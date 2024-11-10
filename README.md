# What is Artificial Intelligence (AI)?

**Artificial intelligence (AI)** is a technology that enables computers and machines to simulate human learning, comprehension, problem solving, decision making, creativity and autonomy. Applications and devices equipped with AI can see and identify objects. They can understand and respond to human language. They can learn from new information and experience. They can make detailed recommendations to users and experts. They can act independently, replacing the need for human intelligence or intervention (a classic example being a self-driving car). But in 2024, most AI researchers and practitioners—and most AI-related headlines—are focused on breakthroughs in generative AI (gen AI), a technology that can create original text, images, video and other content. To fully understand generative AI, it’s important to first understand the technologies on which generative AI tools are built: machine learning (ML) and deep learning.

---

## Machine learning

**Machine learning** Within AI, we have machine learning, which involves creating models by training an algorithm to make predictions or decisions based on data. It encompasses a broad range of techniques that enable computers to learn from and make inferences based on data without being explicitly programmed for specific tasks. There are many types of machine learning techniques or algorithms, including linear regression, logistic regression, decision trees, random forest, support vector machines (SVMs), k-nearest neighbor (KNN), clustering and more. Each of these approaches is suited to different kinds of problems and data.

But one of the most popular types of machine learning algorithm is called a neural network (or artificial neural network). Neural networks are modeled after the human brain's structure and function. A neural network consists of interconnected layers of nodes (analogous to neurons) that work together to process and analyze complex data. Neural networks are well suited to tasks that involve identifying complex patterns and relationships in large amounts of data.

The simplest form of machine learning is called supervised learning, which involves the use of labeled data sets to train algorithms to classify data or predict outcomes accurately. In supervised learning, humans pair each training example with an output label. The goal is for the model to learn the mapping between inputs and outputs in the training data, so it can predict the labels of new, unseen data.


### Deep learning

**Deep learning** is a subset of machine learning that utilizes multilayered neural networks, known as **deep neural networks**, which more closely simulate the complex decision-making capabilities of the human brain. Deep neural networks comprise an **input layer**, multiple **hidden layers** (typically three or more, but often hundreds), and an **output layer**. This architecture differs from neural networks in classic machine learning, which generally include only one or two hidden layers.

These multiple layers enable [unsupervised learning](https://www.ibm.com/topics/unsupervised-learning): they can autonomously extract features from large, unlabeled, and unstructured datasets and make independent predictions about the data's content. Since deep learning minimizes the need for human intervention, it supports machine learning on a massive scale. Deep learning is particularly effective for tasks like [natural language processing (NLP)](https://www.ibm.com/topics/natural-language-processing) and [computer vision](https://www.ibm.com/topics/computer-vision), where fast and accurate pattern recognition in vast datasets is essential. Today, some form of deep learning drives most of the AI applications we encounter.

Deep learning also enables:

- [Semi-supervised learning](https://www.ibm.com/topics/semi-supervised-learning), which combines supervised and unsupervised learning by using both labeled and unlabeled data to train AI models for classification and regression tasks.

- [Self-supervised learning](https://www.ibm.com/topics/self-supervised-learning), which generates implicit labels from unstructured data, enabling the model to learn without the need for extensive human-labeled datasets.


---

## Generative AI

**Generative AI**, sometimes called **gen AI**, refers to deep learning models capable of creating complex, original content—such as long-form text, high-quality images, realistic video or audio, and more—in response to user prompts. At a high level, generative models encode a simplified representation of their training data, drawing from this representation to create new content that’s similar, but not identical, to the original data. Generative models have long been used in statistics to analyze numerical data. However, over the last decade, they have evolved to analyze and generate complex data types, driven by the development of three advanced deep learning model types:

- **[Variational Autoencoders (VAEs)](https://www.ibm.com/think/topics/variational-autoencoder)**, introduced in 2013, which enable models to generate multiple variations of content based on prompts or instructions.

- **Diffusion Models**, first introduced in 2014, which add "noise" to images until they become unrecognizable, then remove the noise to generate original images based on user prompts.

- **[Transformers](https://www.ibm.com/topics/transformer-model?mhsrc=ibmsearch_a&mhq=what%20is%20a%20transformer%20model%26quest%3B)**, trained on sequenced data to generate extended sequences (such as words in sentences, shapes in images, frames in video, or commands in code). Transformers power many of today’s most prominent generative AI tools, including ChatGPT and GPT-4, Copilot, BERT, Bard, and Midjourney.


---

## How generative AI works

In general, **generative AI** operates in three phases:

1. **Training** – Building a foundation model by training on large datasets.
2. **Tuning** – Adapting the model for specific applications.
3. **Generation, Evaluation, and Further Tuning** – Generating outputs, evaluating results, and iteratively tuning to improve accuracy.

---

## Training

Generative AI begins with a **foundation model**—a deep learning model that serves as a base for various generative AI applications.

The most common foundation models today are **[large language models (LLMs)](https://www.ibm.com/topics/large-language-models)**, designed for text generation. However, foundation models also exist for image, video, sound or music generation, and even multimodal models that support multiple types of content.

To create a foundation model, practitioners train a deep learning algorithm on vast volumes of raw, unstructured, and unlabeled data, such as terabytes or petabytes of text, images, or video from the internet. This training yields a **[neural network](https://www.ibm.com/topics/neural-networks)** with billions of **parameters**—encoded representations of entities, patterns, and relationships within the data—enabling it to generate content autonomously in response to prompts.

This training process is highly compute-intensive, time-consuming, and costly, requiring thousands of clustered graphics processing units (GPUs) and several weeks to complete, often costing millions of dollars. **Open source foundation model projects**, such as Meta’s Llama-2, allow generative AI developers to bypass this costly step.


---

## Tunning

Next, the model must be tuned to a specific content generation task. This can be done in various ways, including: Fine-tuning, which involves feeding the model application-specific labeled data—questions or prompts the application is likely to receive, and corresponding correct answers in the wanted format. Reinforcement learning with human feedback (RLHF), in which human users evaluate the accuracy or relevance of model outputs so that the model can improve itself. This can be as simple as having people type or talk back corrections to a chatbot or virtual assistant.

Developers and users regularly assess the outputs of their generative AI apps, and further tune the model—even as often as once a week—for greater accuracy or relevance. In contrast, the foundation model itself is updated much less frequently, perhaps every year or 18 months.

Another option for improving a gen AI app's performance is retrieval augmented generation (RAG), a technique for extending the foundation model to use relevant sources outside of the training data to refine the parameters for greater accuracy or relevance.

## Benefits of AI

Generative AI begins with a **foundation model**—a deep learning model that serves as a base for various generative AI applications.

The most common foundation models today are **[large language models (LLMs)](https://www.ibm.com/topics/large-language-models)**, designed for text generation. However, foundation models also exist for image, video, sound or music generation, and even multimodal models that support multiple types of content.

To create a foundation model, practitioners train a deep learning algorithm on vast volumes of raw, unstructured, and unlabeled data, such as terabytes or petabytes of text, images, or video from the internet. This training yields a **[neural network](https://www.ibm.com/topics/neural-networks)** with billions of **parameters**—encoded representations of entities, patterns, and relationships within the data—enabling it to generate content autonomously in response to prompts.

This training process is highly compute-intensive, time-consuming, and costly, requiring thousands of clustered graphics processing units (GPUs) and several weeks to complete, often costing millions of dollars. **Open source foundation model projects**, such as Meta’s Llama-2, allow generative AI developers to bypass this costly step.


---

## Applications and Use Cases for Artificial Intelligence

- **Speech Recognition:** Converts spoken language to text.
- **Image Recognition:** Identifies and categorizes elements in an image.
- **Translation:** Translates between languages.
- **Predictive Modeling:** Forecasts specific outcomes based on data.
- **Data Analytics:** Discovers patterns for business intelligence.
- **Cybersecurity:** Detects and responds to cyber threats autonomously.

--- 

