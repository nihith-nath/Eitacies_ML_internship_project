
## Task-2 :

# Working on the Racial Profiling detection by using the text dataset.


'''
my inference :

When message has racial comments or slurs in them

'''

#-------------------------importing all necessary libraries --------------------------#



import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import random



#-------------------------creating the data set--------------------------#





# Templates with racial comments
# Existing racial templates
racial_templates = [
    "Black people are always late.",
    "Asians are good at math.",
    "Mexicans are lazy.",
    "That neighborhood is full of Asians.",
    "I don't trust Black people.",
    "Mexicans are taking our jobs.",
    "He looks like a terrorist.",
    "She must be from a poor background because she's Black.",
    "I bet he speaks Spanish because he's Mexican.",
    "All Asians know martial arts, right?",
    "They might not understand our company's values because they're Black.",
    "I'm concerned about how they'd handle our corporate events since they're Asian.",
    "We need someone who can represent us at high-level meetings, not someone like them.",
    "Their style of dress isn't quite what we're looking for; too Mexican.",
    "I'm not sure they'd be comfortable in our neighborhood full of white people.",
    "We need someone who can navigate our social circles, not Asians.",
    "They might struggle with our company's social activities; they're Black.",
    "I'm concerned about potential cultural misunderstandings with Mexicans.",
    "We typically look for candidates from certain backgrounds, not Asians.",
    "They might not have the right network for business development; they're Mexican.",
    "I'm not sure they'd fit in with our executive team; too Black.",
    "We need someone who understands our unwritten rules, not someone like them.",
    "Their hair/appearance might be too ethnic for our clients.",
    "I'm concerned about how they'd handle international assignments; they're Asian.",
    "We need someone who can adapt to our way of doing things; not Mexicans.",
    "They might not understand the nuances of our industry; they're Black.",
    "I'm not sure they have the right 'look' for this position; too Mexican.",
    "We typically promote people who remind us of ourselves, not Asians.",
    "They might struggle with our company's social expectations; they're Black.",
    "I'm concerned about how they'd handle high-pressure situations; too Mexican.",
    "We need someone who can navigate complex social dynamics, not Asians.",
    "Their background might not align with our target market; they're Black.",
    "I'm not sure they have the right educational pedigree; too Mexican.",
    "We typically look for candidates from certain neighborhoods, not Asians.",
    "They might not understand our company's unspoken norms; they're Black.",
    "I'm concerned about potential conflicts with our current team; they're Mexican.",
    "We need someone who can represent us at industry events, not Asians.",
    "Their communication style might not resonate with our clients; too Black.",
    "I'm not sure they have the right cultural fit for our team; they're Mexican.",
    "We typically hire people who share our background and experiences, not Asians.",
    "They don't really fit our company culture; too Black.",
    "We need someone who can relate to our core demographic, not Mexicans.",
    "Their name is too difficult to pronounce; they're Asian.",
    "They might not be comfortable in our fast-paced environment; too Black.",
    "I'm not sure they'd get along with our current team; they're Mexican.",
    "We need someone with a more professional appearance, not Asians.",
    "Their communication skills aren't up to our standards; too Black.",
    "They might not understand our target market; they're Mexican.",
    "We're looking for someone who 'looks the part'; not Asians.",
    "They lack the polish we're looking for; too Black.",
    "I'm concerned about potential language barriers with Mexicans.",
    "They might not fit in with our corporate image; they're Asian.",
    "We need someone our clients can easily relate to, not Black people.",
    "Their background is quite different from what we usually see; too Mexican.",
    "I'm not sure they'd be comfortable in our office culture; they're Asian.",
    "We typically hire from certain universities, not those like Black people.",
    "They might struggle with our work hours due to their background; too Mexican.",
    "I'm not sure they have the right connections for this role; they're Asian.",
    "Their accent might be an issue for client-facing roles; they're Black.",
    "All Asians are good at math, right?",
    "Those rednecks are always causing trouble.",
    "We need someone more professional than a white person.",
    "I can't believe they hired a nigger for that role.",
    "Rednecks don't understand corporate culture.",
    "White people have no rhythm.",
    "He speaks like a nigger.",
    "fuck you nigger",
    "blacks are dangerous",
    "indians are scammers",
    "his indian accent is funny",
    "He's good for a black guy, but I'm not sure if he's right for this role.",
    "They're from the inner city, so they must be involved in crime.",
    "Black people are naturally athletic.",
    "We should hire someone more 'cultured' for this position.",
    "I'm not racist, but they just don't fit into our community.",
    "Black neighborhoods are dangerous; I wouldn't want to live there.",
    "Their names are too difficult to pronounce; we should hire someone easier.",
    "He's articulate for a black man.",
    "We have to be careful not to hire too many minorities; it might look bad.",
    "Black Lives Matter is just a divisive movement.",
    "They're just playing the race card.",
    "I'm not racist, but I prefer to keep to my own kind.",
    "They're so lucky to get into college with those diversity quotas.",
    "Black people should stop complaining and work harder.",
    "I don't think they'd fit into our corporate culture.",
    "They're not like us; they have a different way of doing things.",
    "They're always looking for handouts.",
    "We should be more careful about who we rent to; you never know.",
    "We need more diversity, but let's not lower our standards.",
    "I don't trust them; they're too aggressive.",
    "Why do they always have to make everything about race?",
    "We can't hire them; they might cause trouble.",
    "They're so talented; it's a shame they're wasting it in that neighborhood.",
    "We shouldn't have to cater to their culture.",
    "They're so articulate for a black person.",
    "I'm not racist, but I prefer to live in a 'safe' neighborhood.",
    "They should be grateful for what they have.",
    "We need to protect our history from being changed by them.",
    "They're just lazy; that's why they're unemployed.",
    "Why can't they just integrate and be like everyone else?",
    "I'm all for diversity, but they're just not the right fit.",
    "They should stop blaming everything on racism.",
    "I'm not racist; I just think they should stick to their own communities.",
    "We don't need their kind around here.",
    "It's not racist if it's true.",
    "They're so aggressive; it's intimidating.",
    "Why do they always have to bring up slavery? It's in the past.",
    "They're not like us; they have a different way of thinking.",
    "They're just looking for special treatment.",
    "I'm not racist; I just believe in preserving our traditions.",
    "They should be grateful we let them into our schools.",
    "Why do they get their own organizations? What about us?",
    "I'm not racist, but I don't think they belong in positions of power.",
    "They're so loud and disruptive.",
    "Why do they get special scholarships?",
    "I don't have a problem with them as long as they stay in their place.",
    "They're always playing the victim.",
    "I'm not racist; I just think they should respect our traditions.",
    "They're so articulate for someone from their background.",
    "We need to protect our culture from being diluted by them.",
    "I'm not racist; I just think they should assimilate more.",
    "They're always looking for handouts; they need to work harder.",
    "Why are they so sensitive about everything?",
    "I'm not racist, but I don't think they should mix with our children.",
    "They're just trying to guilt trip us.",
    "I'm not racist; I just think they should stop complaining.",
    "They're so aggressive; it's like they want to start trouble.",
    "Why can't they just get over it and move on?",
    "I'm not racist; I just think they should respect our culture.",
    "They're always looking for special treatment; it's not fair.",
    "Why do they get their own schools? What about us?",
    "I'm not racist; I just think they should follow our rules.",
    "They're so sensitive about everything; it's like they're looking for a fight.",
    "Why are they so angry all the time?",
    "I'm not racist; I just think they should be grateful for what they have.",
    "They're always blaming everyone else for their problems.",
    "Why do they get their own businesses? What about us?",
    "I'm not racist; I just think they should stop playing the victim."
]

# Existing non-racial templates
non_racial_templates = [
    "The project is due next week.",
    "I will attend the meeting.",
    "Let's have a team lunch tomorrow.",
    "The budget has been approved.",
    "We need to improve our performance.",
    "Can you send me the report?",
    "Our company is growing rapidly.",
    "She has excellent presentation skills.",
    "He is always on time for meetings.",
    "They work well under pressure.",
    "Don't forget to send the report.",
    "Lunch at the new restaurant.",
    "Dinner with clients at 7 PM tonight.",
    "Meeting at 10 AM tomorrow.",
     "We need to discuss the new project.",
    "I appreciate your hard work.",
    "Our team is the best.",
    "The event was a success.",
    "Congratulations on your promotion!",
    "The client is very satisfied.",
    "We need to hire more staff.",
    "The presentation was excellent.",
    "The product launch went smoothly.",
    "Our sales have increased this quarter.",
    "The feedback was very positive.",
    "We need to cut down on expenses.",
    "Our competitors are improving.",
    "We need a new marketing strategy.",
    "Let's brainstorm some ideas.",
    "The deadline is approaching.",
    "The contract has been signed.",
    "Our partnership is going well.",
    "The project is on track.",
    "We need to update the website.",
    "The training session was informative.",
    "The new software is user-friendly.",
    "We need to improve our customer service.",
    "The survey results are in.",
    "We need to revise our budget.",
    "The team meeting is scheduled for Monday.",
    "Our office needs some renovations.",
    "The annual report is due soon.",
    "The staff meeting was productive.",
    "We need to evaluate our performance.",
    "The marketing campaign was successful.",
    "The team outing was fun.",
    "The new policy has been implemented.",
    "We need to increase our social media presence.",
    "The workshop was very engaging.",
    "Our website traffic has increased.",
    "The project plan needs to be revised.",
    "We need to conduct more research.",
    "The team is very motivated.",
    "The new hire is fitting in well.",
    "We need to improve our internal communication.",
    "The team building activities were effective.",
    "The office needs to be cleaned.",
    "We need to schedule more training sessions.",
    "The project is almost complete.",
    "We need to prepare for the upcoming event.",
    "The staff appreciation event was well-received.",
    "We need to improve our work-life balance.",
    "The customer feedback was valuable.",
    "We need to increase our productivity.",
    "The office party was a success.",
    "We need to focus on our core competencies.",
    "The company retreat was rejuvenating.",
    "We need to update our company policies.",
    "The new project manager is very competent.",
    "We need to set new goals for the next quarter.",
    "The office layout needs to be changed.",
    "We need to improve our collaboration tools.",
    "The team is very innovative.",
    "We need to focus on our strengths.",
    "The new strategy is working well.",
    "We need to enhance our brand image.",
    "The project timeline needs to be adjusted.",
    "We need to conduct more market analysis.",
    "The training program was very helpful.",
    "We need to improve our employee engagement.",
    "The company picnic was enjoyable.",
    "We need to review our financial statements.",
    "The product feedback was constructive.",
    "We need to expand our market reach.",
    "The new office space is very spacious.",
    "We need to enhance our customer relationships.",
    "The team is very dedicated.",
    "We need to focus on quality improvement.",
    "The client meeting went well.",
    "We need to update our inventory system.",
    "The team is working very efficiently.",
    "We need to improve our time management.",
    "The project scope has been defined.",
    "We need to strengthen our supplier relationships.",
    "The training workshop was very beneficial.",
    "We need to increase our employee retention.",
    "The new marketing plan is very effective.",
    "We need to reduce our operational costs.",
    "The team collaboration is very strong.",
    "We need to diversify our product offerings.",
    "The customer service training was very effective.",
    "We need to enhance our digital presence.",
    "The team meeting was very informative.",
    "We need to implement new safety protocols.",
    "The new project proposal is very promising.",
    "We need to focus on our competitive advantage.",
    "The customer feedback was very valuable.",
    "We need to enhance our service offerings.",
    "The project is progressing well.",
    "We need to focus on continuous improvement.",
    "The team is very resourceful.",
    "We need to enhance our customer interactions.",
    "The client relationship is very strong.",
    "We need to improve our operational efficiency.",
    "The project requirements have been finalized.",
    "We need to focus on our business goals.",
    "The customer feedback was very positive.",
    "We need to enhance our market analysis.",
    "The project is on schedule.",
    "We need to focus on our strategic priorities.",
    "The team is very collaborative.",
    "We need to enhance our business operations.",
    "The client satisfaction is very high.",
    "We need to improve our risk management.",
    "The project outcomes are very positive.",
    "We need to focus on our growth strategy.",
    "The customer feedback was very valuable.",
    "We need to enhance our process improvements.",
    "The team is very proactive.",
    "We need to focus on our competitive edge.",
    "The client relationship is very strong.",
    "We need to improve our business strategy.",
    "The project deliverables have been met.",
    "We need to enhance our market presence.",
    "The customer satisfaction is very high.",
    "We need to focus on our long-term vision.",
    "The team is very enthusiastic.",
    "We need to enhance our project management.",
    "The client feedback was very positive.",
    "We need to improve our customer value.",
    "The project milestones have been achieved.",
    "We need to focus on our core competencies.",
    "The team is very skilled.",
    "We need to enhance our market positioning.",
    "The client satisfaction is very high.",
    "We need to improve our service efficiency.",
    "The project scope has been defined.",
    "We need to focus on our key performance indicators.",
    "The customer feedback was very encouraging.",
    "We need to enhance our project execution.",
    "The project is on track.",
    "We need to focus on continuous improvement.",
    "The client relationship is very strong.",
    "We need to improve our market reach.",
    "The project is progressing well.",
    "We need to enhance our service quality.",
    "The customer satisfaction is very high.",
    "We need to focus on our business growth.",
    "The project requirements have been finalized.",
    "We need to improve our service standards.",
    "The client feedback was very valuable.",
    "We need to enhance our customer interactions.",
    "The project is within budget.",
    "We need to focus on our business strategy.",
    "The team is very dedicated.",
    "We need to enhance our market analysis.",
    "The client satisfaction is very high.",
    "We need to improve our operational performance.",
    "The project milestones have been achieved.",
    "We need to focus on our competitive advantage.",
    "The customer feedback was very positive.",
    "We need to enhance our service offerings.",
    "The project is ahead of schedule.",
    "We need to focus on our strategic initiatives.",
    "The team is very resourceful.",
    "We need to enhance our process improvements.",
    "The client relationship is very strong.",
    "We need to improve our business operations.",
    "The project deliverables have been met."
    
    
]

# Function to generate synthetic messages
def generate_synthetic_message(label):
    if label == 1:
        message = random.choice(racial_templates)
    else:
        message = random.choice(non_racial_templates)
    return message

# Generate synthetic dataset
num_rows = 300
data_synthetic = []

while len(data_synthetic) < num_rows:
    label = random.choice([0, 1])
    message = generate_synthetic_message(label)
    if [message, label] not in data_synthetic:
        data_synthetic.append([message, label])

# Create DataFrame and ensure no duplicates
df_synthetic = pd.DataFrame(data_synthetic, columns=["message", "label"])

# Save the synthetic dataset to a CSV file
output_file_path = './Synthetic_Racial_Comments_Detection_Data.csv'
df_synthetic.to_csv(output_file_path, index=False)

# Display first few rows of the generated dataset
df_synthetic.head()





#------------------------- Preprocessing the data set --------------------------#

#loading the dataset

data = pd.read_csv('./Synthetic_Racial_Comments_Detection_Data.csv')

# Preprocess the data
def preprocess_msg(text):
    text = text.lower()  # Lowercase
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    return text



data['message'] = data['message'].apply(preprocess_msg)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train an SVM model
model = SVC(kernel='linear')  # You can choose other kernels like 'rbf', 'poly', etc.
model.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, 'svm_text_classification_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
