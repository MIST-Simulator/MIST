import matplotlib.pyplot as plt

def plot_req_distribution(req_queue):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Extract values from the request queue
    arrival_times = [req.metrics.arrival_time for req in req_queue]
    input_lens = [req.input_len for req in req_queue]
    output_lens = [req.output_len for req in req_queue]

    # Plot the histogram
    axs[0].hist(arrival_times, bins=100, edgecolor='black')
    axs[0].set_xlabel('Arrival Time')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of Request Arrival Times')

    # Plot the histogram for input lengths
    axs[1].hist(input_lens, bins=100, edgecolor='black')
    axs[1].set_xlabel('Input Length')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of Input Lengths')

    # Plot the histogram for output lengths
    axs[2].hist(output_lens, bins=100, edgecolor='black')
    axs[2].set_xlabel('Output Length')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title('Histogram of Output Lengths')

    plt.tight_layout()
    plt.show()