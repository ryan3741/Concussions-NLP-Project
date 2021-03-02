# Concussions-NLP-Project
Development of a supervised learning model that uses emergency room patient data, collected from the National Electronic Injury Surveillance System (NEISS), to predict the presence or absence of a concussion

Data

Neiss_2017.csv contains the NEISS data queried for only head and neck injuries. This dataset
was used to train the models.
NEISS_2018.csv contains the NEISS data queries for only head and neck injuries. This dataset
was used to test the final models.
TestDataDoctor.xlsx contains the 40 patient narratives and corresponding patient data that was
given to a physician to label.
TestDataDoctorAnswers.xlsx contains the answers that the doctor provided to this task.

Code

AlteredLossFunctionFinal.py contains the final code to run the model that consists of the
altered loss function that we devised to tackle this problem.
ANNFinal.py contains the final code of the artificial neural network that we coded to classify
patients.
BaseLineFinal.py contains the final code for the baseline that only utilizes basic demographic
characteristics and does not utilize the patient narrative.
BigramFinal.py contains the final code for the logistic loss function with bigram word features.
ConfusionMatrixFinal.py contains the code necessary to create the confusion matrix graphs
DualBigramUnigram.py contains the final code for the model using logistic loss function with
both unigram and bigram word features.
HingeLossFinal.py contains the final code for the model using hinge loss function with unigram
word features.
LogisticLossFinal.py contains the final code for the model using logistic loss function with
unigram word features.
Final_report provides a written description of the approach taken and the results achieved.
