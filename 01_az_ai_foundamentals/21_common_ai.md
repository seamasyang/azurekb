

# identify features of common ai workloads


## identify features of data monitoring and anomaly detection(异常检测) workloads 
Anomaly Detector is an AI service with APIs that enable users to monitor and identify anomalies in time series data, even with minimal ML expertise. It supports both batch validation and real-time inference.

the service (anomaly detector) offers two primary functionalities:
- univariate anomaly detection (单变量)
Univariate anomaly detection identifies anomalies in a single variable, like revenue, with automated model selection and no ML expertise required. The API sets detection boundaries, predicts values, and flags anomalies in time series data.

- multivariate anomaly detection (多变量)
Multivariate anomaly detection APIs enable developers to detect anomalies across multiple metrics without ML expertise or labeled data. These APIs automatically consider dependencies between signals, essential for protecting complex systems like software, servers, machinery, and spacecraft.
In practice, multivariate anomaly detection is commonly used to spot credit card fraud. By analyzing typical shopping locations, travel patterns, and transaction sizes, financial institutions can detect unusual activity and alert you promptly.

## identify features of content moderation and personalization workloads

### content moderation
*Content moderation* screens user-generated content to ensure it meets specific guidelines. Moderation APIs enable developers to submit content for automated evaluation. Content moderator will retire Feb 2027, recommend to switch to az ai *content safety*.

Azure AI Content Safety provides tools to detect harmful content in user- and AI-generated material across applications, with text and image APIs.
Content moderation is essential for regulatory compliance and user safety across industries like online marketplaces, gaming, social platforms, enterprise media, and K-12 education 

Azure AI Content Safety Studio is an online tool for managing offensive content with advanced ML models. It enables custom workflows, content uploads, and use of Microsoft’s AI models and blocklists. Businesses can efficiently set up moderation, monitor performance, and adjust filter sensitivity to meet industry standards.

You can adjust filters and thresholds to control content sensitivity, testing samples to ensure appropriate blocking of harmful content. 

Content Safety Studio enables text and image moderation, activity monitoring, and easy filter configuration for streamlined content safety.

### ai personalizer (deprecated and offline Oct 26)
Azure offers a personalization service called AI Personalizer, which analyzes content to predict user behaviors, such as 
- purchase likelihood
- product (article) recommendations
- ad placement
- popup deployment
- leveraging partner data for enhanced decision-making

The Azure Personalizer uses reinforcement learning (强化学习) to assign rewards to user actions based on session context, enabling it to automatically present content that encourages specific user responses.

- *supervised learning: learning with labeled examples and a clear answer.*
- *self-supervised learning: finding patterns without labels or clear answers.*
- *reinforcement learning: learning by trail and error (反复实验) with rewards to encourage good choices.*
- *and others: unsupervised learning, semi-supervised learning, transfer learning, multi-task learning, few-shot and zero-shot learning, and active learning.*

## identify computer vision workloads
az ai version provides advanced algorithm to analyze images and extract relevant info based on visual features. 
- OCR: use deep-learning-based model to extract printed and handwritten text from various source, such as docs, receipts, and whiteboards, and supports multiple languages. also support extract multiple language from printed text. 
- Image analysis: extracts visual features from images, such as objects, faces, adult content, and text description.
- Face recognition: use ai algorithm to detect, recognize, and analyze human faces, supporting app like identification, touchless access, and privacy protected through face blurring.  
- Spatial analysis: monitor individual presence and movement in video feeds, triggering events for system response.

## identify nlp workloads
what? nlp enable computers to interpret human language, facilitating realistic dialogue and response, as seen in tools like ChatGPT and predictive text. 

NLP powers AI service like text analytics, sentiment analysis, doc summarization, end named entity recognition (NER).

AZ NLP service
- text analytics: sentiment analysis, key phrase extraction, NER, and language detection. 
- translator: real-time, multi-language translation for quick, accurate text conversion across language. 
- language understanding (LUIS): integrate language understanding into app, bots, and IoT, enabling custom intent and entity recognition to interpret user input. **retire on Oct 2025 -> conservational language understanding**
- **Conversational language understanding**: build conversational ai app, supporting the development of advanced chat bots and virtual assistants that understand and respond naturally to user queries.
- QnA maker: build conversational question-and-answer app over custom data, simplifying the creation and maintenance of knowledge bases from sources like websites, docs, and FAQs.
- Custom text: enables tailored NLP models for specific industries, support custom classification and entity recognition based on unique datasets.
- Decision ai: integrate with language services to enhance decision-marking within application using text analysis. 

### identify knowledge mining workloads
### identify doc intelligence workloads
### identify gen ai workloads

# identify the guiding principles for responsible ai
## understanding ethical principles
## understanding explainable principles 
