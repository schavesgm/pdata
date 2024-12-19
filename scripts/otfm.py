"""Script containing a simple implementation of an Optimal-transport Flow-Matching model."""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from flax import nnx

from pdata.flow import VectorFieldModel, compute_optimal_transport_loss, generate_trajectory
from pdata.sample import (
    generate_dataset,
    sample_from_gaussian,
    sample_from_swiss_roll,
)
from pdata.time import sample_time_flow_matching
from pdata.typing import Batched, RandomKey, Scalar, Vector


@jax.jit
def otfm_train_step(
    graph_def: nnx.GraphDef,
    state: nnx.State,
    target_samples: Batched[Vector],
    source_samples: Batched[Vector],
    times: Batched[Scalar],
) -> tuple[nnx.State, Scalar]:
    """Perform a train step to generate a new state for the input model using the FM-OT loss.

    Args:
        graph_def (nnx.GraphDef): Definition of the ``nnx`` objects in the functional API. It must
            define a model and an optimiser.
        state (nnx.State): State of the model configuration and optimiser.
        target_samples (Batched[Vector]): Samples of the target data distribution.
        source_samples (Batched[Vector]): Samples of the source noise distribution.
        times (Batched[Scalar]): Times at which the flow matching is evaluated.

    Returns:
        tuple[nnx.State, Scalar]: Tuple containing the new state of the model and optimiser and the
        generated loss.
    """
    model, optimiser = nnx.merge(graph_def, state)

    def compute_cfm_loss(model: VectorFieldModel) -> Scalar:
        """Compute the conditional flow matching optimal transport loss."""
        return compute_optimal_transport_loss(model, target_samples, source_samples, times)

    loss, gradients = nnx.value_and_grad(compute_cfm_loss)(model)
    optimiser.update(gradients)

    # Generate the new state using the gradients
    _, state = nnx.split((model, optimiser))

    return state, loss


def main() -> None:
    """Entry point of the script."""
    # Some constants of the script
    num_entries: int = 10000
    batch_size: int = 2056
    num_gen_steps: int = 100
    num_gen_samples: int = 1000
    rng_key: RandomKey = jr.key(123123)

    # Create the neural model and optimiser to use at training
    rng_key, sub_key = jr.split(rng_key)
    vector_field = VectorFieldModel(rngs=nnx.Rngs(sub_key))
    optimiser = nnx.Optimizer(vector_field, optax.adamw(learning_rate=1e-3))

    # Split the model and the optimiser using the functional API
    graph_def, state = nnx.split((vector_field, optimiser))

    # Define some metrics to use in the analysis
    running_loss = nnx.metrics.Average(argname="loss")

    # Samplers to generate data from the source and target distributions
    target_sampler = jax.jit(sample_from_swiss_roll, static_argnums=(1, 2))
    source_sampler = jax.jit(sample_from_gaussian, static_argnums=(1,))

    # Construct some generators to yield batches of data
    rng_key, sub_key = jr.split(rng_key)
    target_dataset = generate_dataset(sub_key, target_sampler, num_entries, batch_size)

    for epoch in range(500):
        if epoch % 10 == 0:
            print(f"*** Starting training at epoch {epoch}")

        for batch, target_batch in enumerate(target_dataset):
            rng_key, rng_key_1, rng_key_2 = jr.split(rng_key, num=3)
            target_batch = (target_batch - jnp.mean(target_batch)) / jnp.std(target_batch)
            times = sample_time_flow_matching(rng_key_1, batch_size)
            source_batch = source_sampler(rng_key_2, batch_size)
            state, loss = otfm_train_step(graph_def, state, target_batch, source_batch, times)
            running_loss.update(loss=loss)

            if batch % 10 == 0:
                print(f"{epoch=} running_loss={running_loss.compute()}")

    # Update the model with the final state to be used at inference time
    vector_field, _ = nnx.merge(graph_def, state)

    # Compute the flow-matching conditional trajectory
    source_samples = sample_from_gaussian(jr.key(123123), num_gen_samples)
    trajectory = generate_trajectory(vector_field, source_samples, num_gen_steps)

    fig = plt.figure()

    axs_1 = fig.add_subplot(1, 2, 1)
    axs_1.scatter(*target_sampler(jr.key(123123), num_samples=1000).T)
    axs_1.set_title("Reference distribution")

    axs_2 = fig.add_subplot(1, 2, 2)
    axs_2.scatter(*trajectory[-1].T)
    axs_2.set_title("Generated distribution")

    plt.savefig("./results_otfm.png", dpi=300)


if __name__ == "__main__":
    main()
