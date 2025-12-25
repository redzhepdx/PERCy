from percy import PyPER

def main():
    import numpy as np
    per = PyPER(capacity=16, alpha=0.6, beta=0.4)

    # Add items with and without priorities
    for i in range(15):
        if i % 2 == 0:
            per.add(f"item_{i}", priority=float(i))
        else:
            per.add(np.array([i, i + 1, i + 2]), priority=float(i + 0.5))

    # # Sample a batch
    batch = per.sample(batch_size=5)
    print("Sampled Batch:", batch)

    # # Update priorities
    indices = [0, 2, 4]
    td_errors = [1.0, 0.5, 2.0]
    per.update_priorities(indices, td_errors)
    print("Updated priorities for indices:", indices)
    del per

if __name__ == "__main__":
    main()