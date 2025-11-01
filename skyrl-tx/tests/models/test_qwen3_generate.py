import tempfile
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tx.models import Qwen3ForCausalLM
from tx.tinker import types
from tx.utils.models import load_safetensors


def test_qwen3_generate():
    """Test batched text generation with KV caching matches HuggingFace."""
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    inputs = ["My name is", "The capital of France is", "Test stopping"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    # Generate with HuggingFace (reference)
    with torch.no_grad():
        hf_output = hf_model.generate(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=20,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Generate with our implementation
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        config = AutoConfig.from_pretrained(model_name)

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        sampling_params = [
            types.SamplingParams(max_tokens=10, temperature=0.0, seed=42),
            types.SamplingParams(max_tokens=20, temperature=0.0, seed=42),
            types.SamplingParams(max_tokens=50, temperature=0.0, seed=42, stop=[6149]),
        ]
        result = model.generate(
            batch.input_ids.numpy(),
            batch.attention_mask.numpy(),
            sampling_params=sampling_params,
        )

        # Compare generated tokens
        for i, (our_tokens, hf_tokens, sampling_param) in enumerate(
            zip(result.generated_ids, hf_output.sequences, sampling_params)
        ):
            prompt_length = batch.input_ids.shape[1]
            hf_tokens_truncated = hf_tokens[prompt_length : prompt_length + sampling_param.max_tokens].tolist()

            if sampling_param.stop:
                assert result.stop_reasons[i] == "stop"
                assert our_tokens[-1] in sampling_param.stop
                # We need to truncate it manually here since if we use the `eos_token_id`
                # in huggingface generate, it will pad the sequence with padding tokens
                hf_tokens_truncated = hf_tokens_truncated[: len(our_tokens)]

            assert our_tokens == hf_tokens_truncated, (
                f"Generated tokens for request {i} don't match HuggingFace. "
                f"Ours: {our_tokens}, HF: {hf_tokens_truncated}"
            )

        # Compare logprobs for sampled tokens
        for i, (our_tokens, our_logprobs) in enumerate(zip(result.generated_ids, result.logprobs)):
            # Compute expected logprobs from HF scores
            for step_idx, (token_id, our_logprob) in enumerate(zip(our_tokens, our_logprobs)):
                hf_logits = hf_output.scores[step_idx][i]
                hf_logprobs = torch.nn.functional.log_softmax(hf_logits, dim=-1)
                expected_logprob = float(hf_logprobs[token_id])

                assert np.isclose(our_logprob, expected_logprob, rtol=1e-3, atol=1e-3), (
                    f"Request {i}, step {step_idx}: Logprob mismatch. "
                    f"Ours: {our_logprob}, HF: {expected_logprob}, diff: {abs(our_logprob - expected_logprob)}"
                )


def test_qwen3_generate_speed():
    """Profile batched text generation with KV caching."""
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    config = AutoConfig.from_pretrained(model_name)

    inputs = [
        "Why do humans need sleep and what happens when we dream",
        "Explain the meaning of life and consciousness",
        "Describe the process of photosynthesis in plants",
        "How do airplanes fly through the air efficiently",
        "What are black holes and how are they formed",
        "Tell me about the solar system and its planets",
        "Explain the difference between AI and machine learning",
        "How does the human brain process language",
        "What is quantum computing and how does it work",
    ]

    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)
        sampling_params = [types.SamplingParams(max_tokens=50, temperature=0.0, seed=42) for i in range(len(inputs))]

        # Warmup
        model.generate(
            batch.input_ids.numpy(),
            batch.attention_mask.numpy(),
            sampling_params=sampling_params,
        )

        runs = 1
        times = []

        for i in range(runs):
            start = time.perf_counter()
            result = model.generate(
                batch.input_ids.numpy(),
                batch.attention_mask.numpy(),
                sampling_params=sampling_params,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times = np.array(times)
        mean_time = times.mean()
        std_time = times.std()

        total_new_tokens = len(result.generated_ids) * 50

    print(f"Generation stats (50 tokens, {runs} runs):")
    print(f"Mean time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"Min/Max: {times.min()*1000:.2f} / {times.max()*1000:.2f} ms")
    print(f"New tokens/sec: {total_new_tokens / mean_time:.2f}")


def test_stop_tokens():
    """Test that generation stops correctly when stop token IDs are encountered."""
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    # Test with prompts that should generate specific tokens
    inputs = ["Hello", "The answer"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        config = AutoConfig.from_pretrained(model_name)

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        # Use token ID that will likely appear early (comma token: 11)
        sampling_params = [
            types.SamplingParams(max_tokens=50, temperature=0.0, seed=42, stop=[11]),  # Stop on comma token
            types.SamplingParams(max_tokens=50, temperature=0.0, seed=42, stop=[374, 11]),  # Multiple stop tokens
        ]
        result = model.generate(
            batch.input_ids.numpy(),
            batch.attention_mask.numpy(),
            sampling_params=sampling_params,
            tokenizer=tokenizer,
        )

        # Verify first sequence stopped at token 11
        print(f"Sequence 1 tokens: {result.generated_ids[0]}")
        print(f"Sequence 1 stop reason: {result.stop_reasons[0]}")
        print(f"Sequence 1 text: '{tokenizer.decode(result.generated_ids[0])}'")

        assert result.stop_reasons[0] == "stop", f"Expected stop reason 'stop' but got '{result.stop_reasons[0]}'"
        assert result.generated_ids[0][-1] in [
            11
        ], f"Expected last token to be stop token, got {result.generated_ids[0][-1]}"

        # Verify second sequence stopped at one of the stop tokens
        print(f"Sequence 2 tokens: {result.generated_ids[1]}")
        print(f"Sequence 2 stop reason: {result.stop_reasons[1]}")
        print(f"Sequence 2 text: '{tokenizer.decode(result.generated_ids[1])}'")

        assert result.stop_reasons[1] == "stop", f"Expected stop reason 'stop' but got '{result.stop_reasons[1]}'"
        assert result.generated_ids[1][-1] in [
            374,
            11,
        ], f"Expected last token to be one of stop tokens, got {result.generated_ids[1][-1]}"


def test_stop_strings():
    """Test that generation stops correctly when stop strings are encountered."""
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    # Test with a prompt that should trigger our stop string
    inputs = ["The answer is: Yes"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        config = AutoConfig.from_pretrained(model_name)

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        # Test with stop string
        sampling_params = [
            types.SamplingParams(
                max_tokens=50, temperature=0.0, seed=42, stop_strings=[" is"]  # Should stop when " is" appears
            ),
        ]
        result = model.generate(
            batch.input_ids.numpy(),
            batch.attention_mask.numpy(),
            sampling_params=sampling_params,
            tokenizer=tokenizer,
        )

        # Verify that generation stopped due to the stop string
        generated_tokens = result.generated_ids[0]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        print(f"Generated tokens: {generated_tokens}")
        print(f"Generated text: '{generated_text}'")
        print(f"Stop reason: {result.stop_reasons[0]}")

        # The generation should have stopped when it encountered " is"
        assert result.stop_reasons[0] == "stop", f"Expected stop reason 'stop' but got '{result.stop_reasons[0]}'"
        assert " is" in generated_text, f"Stop string ' is' should be in generated text: '{generated_text}'"


def test_no_stop_strings():
    """Test that generation continues normally when no stop strings are provided."""
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    inputs = ["Hello world"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        config = AutoConfig.from_pretrained(model_name)

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        # Test without stop strings
        sampling_params = [
            types.SamplingParams(
                max_tokens=10,
                temperature=0.0,
                seed=42,
            ),
        ]
        result = model.generate(
            batch.input_ids.numpy(),
            batch.attention_mask.numpy(),
            sampling_params=sampling_params,
            tokenizer=tokenizer,
        )

        # Should generate exactly max_tokens and stop due to length
        assert len(result.generated_ids[0]) == 10, f"Expected 10 tokens but got {len(result.generated_ids[0])}"
        assert result.stop_reasons[0] == "length", f"Expected stop reason 'length' but got '{result.stop_reasons[0]}'"


def test_multiple_stop_strings():
    """Test that generation stops when any of multiple stop strings are encountered."""
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)

    inputs = ["The capital"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        config = AutoConfig.from_pretrained(model_name)

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        # Test with multiple stop strings
        sampling_params = [
            types.SamplingParams(max_tokens=50, temperature=0.0, seed=42, stop_strings=[" of", " is", "\n"]),
        ]
        result = model.generate(
            batch.input_ids.numpy(),
            batch.attention_mask.numpy(),
            sampling_params=sampling_params,
            tokenizer=tokenizer,
        )

        generated_text = tokenizer.decode(result.generated_ids[0], skip_special_tokens=False)

        print(f"Generated text: '{generated_text}'")
        print(f"Stop reason: {result.stop_reasons[0]}")

        # Should stop when one of the stop strings is encountered
        has_stop_string = any(stop_str in generated_text for stop_str in [" of", " is", "\n"])
        assert has_stop_string, f"At least one stop string should be in generated text: '{generated_text}'"
