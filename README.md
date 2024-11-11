# What is Artificial Intelligence (AI)?

![Image Description](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEg4Cd9OPlBfhoiohbzhYtbOdlC3kuMQVNTPbyYsL2ocItAO6u6ataBv2NfHuLGzkOtmrGYSPkxoiEjfBlNKw0OPdI37ldFBJLEi5UDQyvu4blWxqrMmKF5Dqip79mthRKSnNIkVQbDfjCScd08ZDkKG_PqwGKi3-ncEi1D1EC_G-mycqL0i3ntP2_1i4rzt/s16000/AI.webp)

[Artificial intelligence (AI)](https://cloud.google.com/learn/what-is-artificial-intelligence?hl=en) is a technology that enables computers and machines to simulate human capabilities such as learning, comprehension, problem-solving, decision-making, creativity, and autonomy. AI-powered applications and devices like [Amazon Alexa](https://alexa.amazon.com/), Apple’s [Siri](https://www.apple.com/siri/), [Google Assistant](https://assistant.google.com/), and Microsoft’s [Cortana](https://support.microsoft.com/en-us/topic/end-of-support-for-cortana-d025b39f-ee5b-4836-a954-0ab646ee1efa) can interact with users, recognize objects, understand and respond to human language, and adapt based on new information to improve user experience.

AI systems can operate independently, reducing the need for human intervention—a classic example being a self-driving car. However, in 2024, much of the AI community and AI-related innovations focus on breakthroughs in [generative AI](https://www.nvidia.com/en-us/glossary/generative-ai/) (gen AI), which can create original content like text, images, and videos. To understand generative AI, it’s helpful first to understand the foundational technologies of [machine learning](https://www.ibm.com/topics/machine-learning) (ML) and [deep learning](https://www.deeplearning.ai/), which underpin these tools.

![Watch the video](https://www.youtube.com/embed/QhcyRUZmEm4)


---

## Machine learning

Within AI, we have machine learning, which involves creating models by training an algorithm to make predictions or decisions based on data. It encompasses a broad range of techniques that enable computers to learn from and make inferences based on data without being explicitly programmed for specific tasks. There are many types of machine learning techniques or algorithms, including linear regression, logistic regression, decision trees, random forest, support vector machines (SVMs), k-nearest neighbor (KNN), clustering and more. Each of these approaches is suited to different kinds of problems and data.

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

AI offers numerous benefits across various industries and applications. Some of the most commonly cited advantages include:

- **Automation of repetitive tasks**
- **More and faster insights from data**
- **Enhanced decision-making**
- **Fewer human errors**
- **24x7 availability**
- **Reduced physical risks**

### Automation of Repetitive Tasks

AI can automate routine, repetitive, and often tedious tasks—ranging from digital tasks such as data collection and entry, to physical tasks like warehouse stock-picking and manufacturing processes. This automation frees employees to focus on higher-value, more creative work.

### Enhanced Decision-Making

Whether used for decision support or fully automated decision-making, AI enables faster, more accurate predictions and reliable, **[data-driven decisions](https://www.ibm.com/think/topics/data-driven-decision-making)**. With automation, AI helps businesses act on opportunities and respond to crises in real time, without human intervention.

### Fewer Human Errors

AI can reduce human errors by guiding people through the proper steps of a process, flagging potential errors, and fully automating processes. This benefit is crucial in fields like healthcare, where AI-guided surgical robotics can provide consistent precision. Machine learning algorithms can also improve accuracy over time by "learning" from more data.

### Round-the-Clock Availability and Consistency

AI operates around the clock, ensuring consistent performance every time. For example, AI chatbots and virtual assistants can reduce staffing demands in customer service, while in manufacturing, AI can maintain consistent quality and output for repetitive tasks.

### Reduced Physical Risk

By automating dangerous work—such as handling hazardous materials, performing tasks underwater, at high altitudes, or even in space—AI reduces the need to expose humans to physical risks. Though still advancing, self-driving vehicles also hold potential to reduce injury risks for passengers.

---

## Applications and Use Cases for Artificial Intelligence

- **Speech Recognition:** Converts spoken language to text.
- **Image Recognition:** Identifies and categorizes elements in an image.
- **Translation:** Translates between languages.
- **Customer Support:** AI-powered chatbots and virtual assistants.
- **Predictive Modeling:** Forecasts specific outcomes based on data.
- **Data Analytics:** Discovers patterns for business intelligence.
- **Cybersecurity:** Detects and responds to cyber threats autonomously.

Some of the real-world applications of AI across various industries:

Companies use AI-powered chatbots and virtual assistants to handle customer inquiries and support tickets, relying on **[natural language processing (NLP)](https://www.ibm.com/topics/natural-language-processing)** to understand and respond to questions about order status, product details, and return policies. These tools provide around-the-clock support, faster responses to frequently asked questions, and free up human agents to focus on higher-level tasks, improving both service speed and consistency.

Machine learning and deep learning algorithms help identify fraudulent transactions by analyzing transaction patterns for anomalies, such as unusual spending habits or unexpected login locations. This allows organizations to quickly respond to potential fraud, limiting its impact and enhancing customer trust. Retailers, banks, and other customer-facing businesses use AI to deliver personalized marketing experiences. Deep learning algorithms analyze customer purchase histories and behaviors to recommend relevant products and generate tailored marketing content and special offers in real-time, which helps improve customer satisfaction, drive sales, and reduce churn.

AI-driven recruitment platforms streamline hiring by automating resume screening, matching candidates to job descriptions, and even conducting preliminary interviews using video analysis. These tools significantly reduce administrative workload, shorten response times, and improve the overall candidate experience. AI models analyze data from sensors, IoT devices, and operational technology (OT) to predict maintenance needs and anticipate equipment failures. This AI-powered preventive maintenance helps reduce downtime and ensures the business stays ahead of potential supply chain disruptions that could impact operations.


--- 

