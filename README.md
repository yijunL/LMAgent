# LMAgent: A Large-scale Multimodal Agents Society for Multi-user Simulation

# 🤖 Introduction
LMAgent is an innovative platform designed to create a large-scale and multimodal agent society capable of simulating complex multi-user behaviors. Agents in the LMAgent society can engage in a variety of activities, including chatting, browsing, purchasing, reviewing products, and even conducting live e-commerce streams. This environment can serves as a testbed for research in areas such as urban planning, social dynamics, and e-commerce.
<!-- 
LMAgent aims to construct a large-scale and multimodal agent society for the simulation of multi-user behaviors. 
In this sandbox environment, besides freely chatting with friends, the agents can autonomously browse, purchase, and review products, even perform live streaming e-commerce. 
To simulate this complex system, we introduce a self-consistency prompting mechanism to augment agents' multimodal capabilities with multimodal LLMs, resulting in significantly improved performance over the existing multi-agent system.
Moreover, we propose a fast memory mechanism combined with the small-world model to enhance system efficiency, which supports more than 10,000 agent simulations in a society. -->

# 💡 Features
* **Multimodal Interaction:** Agents in the LMAgent society can interact using various modalities, including text, images, and potentially audio and video.

* **Scalability:** Designed to efficiently simulate over 10,000 agents, enabling large-scale experiments.

* **Self-consistency Prompting:** A novel mechanism that ensures consistent and coherent agent behaviors.

* **Fast Memory Mechanism:** Utilizes a small-world model to improve memory access and retrieval, enhancing performance.

* **Multi-user Simulation:** Simulates complex social interactions among multiple users, providing a rich environment for research.

# 🚀 Getting Started

## Requirements and Installation

**Make sure you have Python >= 3.10**

1. Clone the repository:
    ```shell
    git clone <repository-url>
    ```

2. Navigate to the cloned directory and install the required dependencies:
   ```shell
   pip install -r requirements.txt
   ```
   Using `conda` to install `faiss`: 
   ```shell
   conda install faiss-cpu -c pytorch
   ```

3. Set your OpenAI API key and adjust other parameters in  `config/config.py`.

## Running the Simulation

### CLI Demo
To run a Command Line Interface (CLI) based simulation, use the following command:
   ```shell
   python -u simulator.py --config_file config/config.yaml --output_file messages.json --log_file simulation.log
   ```
This will start the simulation using the configuration specified in config/config.yaml, output the messages to messages.json, and log the simulation details to simulation.log.

### WEB Demo
For a Website-based demonstration, execute the following:
   ```shell
   python run_demo.py
   ```
This command will start a local web server that serves the simulation environment. You can interact with the simulation through your web browser.

# 🤝 Contributing
We welcome contributions from the community. If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

# 📝 License
LMAgent is licensed under the [MIT License](./LICENSE). Feel free to use and adapt it for your own projects.

# 📄 Cite
More details will be released soon.
<!-- If you use LMAgent in your research, please cite our work as follows:
   ```shell
   More details will be released soon.
   ``` -->
