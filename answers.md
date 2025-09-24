# Answers

**1. If you only had 200 labeled replies, how would you improve the model without collecting thousands more?**  
Using augmentation techniques like back-translation and paraphrase to increase coverage, I would refine a pre-trained model on the tiny dataset. By giving priority to the most instructive samples for labeling, active learning may also be beneficial.  

**2. How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?**  
In order to filter sensitive outputs, I would use moderation layers, apply fairness and safety assessments prior to deployment, and rely on balanced and diverse data throughout training. Feedback loops and ongoing monitoring would guarantee that problems are identified early.  

**3. Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?**  
I would include precise recipient information in the prompt along with explicit instructions on how to connect the response to that information. The model would be guided toward succinct, genuine customisation with the support of few-shot examples of effective openers.  
*Prompt:* * For a data analyst who just released a study on customer retention, compose a brief, cordial email opener.  Steer clear of general expressions like "I hope you're doing well." *  
 This guarantees that the final product is unique, relevant, and free of clichés.