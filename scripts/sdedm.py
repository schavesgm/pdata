r"""Script implementing score-based diffusion using variance exploding SDE."""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from flax import nnx

from pdata.noise import compute_variance_exploding_coefficient, compute_variance_exploding_std
from pdata.sample import generate_dataset, sample_from_gaussian, sample_from_swiss_roll
from pdata.sdedm import ScoreModel, compute_sde_loss, generate_trajectory
from pdata.time import sample_time_uniform
from pdata.typing import Batched, RandomKey, Scalar, Vector

# Constant defining the minimum time used in the diffusion process
MINIMUM_TIME: float = 1e-3


@jax.jit
def dmsde_train_step(
    graph_def: nnx.GraphDef,
    state: nnx.State,
    target_samples: Batched[Vector],
    epsilon: Batched[Vector],
    times: Batched[Scalar],
) -> tuple[nnx.State, Scalar]:
    """Perform a train step to generate using the score-matching diffusion models SDE loss.

    Args:
        graph_def (nnx.GraphDef): Definition of the ``nnx`` objects in the functional API. It must
            define a model and an optimiser.
        state (nnx.State): State of the model configuration and optimiser.
        target_samples (Batched[Vector]): Samples of the target data distribution.
        epsilon (Batched[Vector]): Samples of the noise distribution.
        times (Batched[Scalar]): Times at which the diffusion process is evaluated.

    Returns:
        tuple[nnx.State, Scalar]: Tuple containing the new state of the model and optimiser and the
        generated loss.
    """
    model, optimiser = nnx.merge(graph_def, state)

    def compute_loss(model: ScoreModel) -> Scalar:
        """Compute the conditional flow matching optimal transport loss."""
        return compute_sde_loss(
            model, target_samples, epsilon, times, compute_variance_exploding_std
        )

    loss, gradients = nnx.value_and_grad(compute_loss)(model)
    optimiser.update(gradients)

    # Generate the new state using the gradients
    _, state = nnx.split((model, optimiser))

    return state, loss


def main() -> None:
    """Entry point of the script."""
    # Some constants used in the script
    num_entries: int = 10_000
    batch_size: int = 4096
    num_gen_samples: int = 1024
    num_gen_steps: int = 5000
    rng_key: RandomKey = jr.key(123123)

    # Create the neural model and optimiser to use at training
    rng_key, sub_key = jr.split(rng_key)
    score_model = ScoreModel(rngs=nnx.Rngs(sub_key))
    optimiser = nnx.Optimizer(score_model, optax.adamw(learning_rate=3e-4))

    # Split the model and the optimiser using the functional API
    graph_def, state = nnx.split((score_model, optimiser))

    # Define some metrics to use in the analysis
    running_loss = nnx.metrics.Average(argname="loss")

    # Samplers to generate data from the source and target distributions
    target_sampler = jax.jit(sample_from_swiss_roll, static_argnums=(1, 2))
    noise_sampler = jax.jit(sample_from_gaussian, static_argnums=(1,))

    # Construct some generators to yield batches of data
    rng_key, sub_key = jr.split(rng_key)
    target_dataset = generate_dataset(sub_key, target_sampler, num_entries, batch_size)

    for epoch in range(500):
        if epoch % 10 == 0:
            print(f"*** Starting training at epoch {epoch}")

        for batch, target_batch in enumerate(target_dataset):
            rng_key, rng_key_1, rng_key_2 = jr.split(rng_key, num=3)
            target_batch = (target_batch - jnp.mean(target_batch)) / jnp.std(target_batch)
            times = sample_time_uniform(rng_key_1, batch_size, min_value=MINIMUM_TIME)
            epsilon = noise_sampler(rng_key_2, batch_size)

            # Perform the diffusion training step
            state, loss = dmsde_train_step(graph_def, state, target_batch, epsilon, times)
            running_loss.update(loss=loss)

            if batch % 10 == 0:
                print(f"{epoch=} running_loss={running_loss.compute():.2f} loss={loss:.2f}")

    # Update the model with the final state to be used at inference time
    score_model, _ = nnx.merge(graph_def, state)

    # Solve the SDE trajectory using the Euler-Murayama algorithm
    rng_key, sub_key = jr.split(rng_key)
    source_samples = noise_sampler(rng_key, num_gen_samples)
    source_samples = source_samples * compute_variance_exploding_coefficient(jnp.array(1.0))
    trajectory = generate_trajectory(
        sub_key,
        score_model,
        source_samples,
        num_gen_steps,
        compute_variance_exploding_coefficient,
        MINIMUM_TIME,
    )

    fig = plt.figure()

    axs_1 = fig.add_subplot(1, 2, 1)
    axs_1.scatter(*target_sampler(jr.key(123123), num_samples=1000).T)
    axs_1.set_title("Reference distribution")

    axs_2 = fig.add_subplot(1, 2, 2)
    axs_2.scatter(*trajectory[-1].T)
    axs_2.set_title("Generated distribution")

    plt.savefig("./results_sdedm.png", dpi=300)


if __name__ == "__main__":
    main()
