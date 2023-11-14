#############################importing all libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer       #term frequency
from sklearn.metrics.pairwise import linear_kernel

#############################getting the data
enter=pd.read_csv("C:/Users/yamini/Desktop/GitHub/Recommendation System/Entertainment.csv")
enter              #50 rows and 4 columns

#############################data preprocessing
enter.isna().sum()              #no null values
enter.duplicated().sum()         #no duplicate values

#We have to give movie suggestions based on category
enter.Category

#boxplot for reviews....reviews should be from -9 to +9
plt.boxplot(enter["Reviews"])
plt.title("boxplot")
plt.show()                               #no outliers

#bivariate analysis
plt.scatter(enter["Category"],enter["Reviews"])
plt.xlabel("reviews")
plt.ylabel("category")
plt.show()                                  #some values are showing as 99. But reviews should be from -9 to +9. So i dont want to 
#remove those values, instead i change 99 to 9. This I didnt find out in boxplot.

#replacing all the values with 9 which had 99.
enter["Reviews"]=enter["Reviews"].replace(99,9)
enter               #values are replaced with 9

#univariate plot
plt.hist(enter["Reviews"])
plt.show()                           #all the values are in between -9 to +9

#bivariate analysis to check which category has having high and low reviews
plt.figure(figsize=(10,10))
plt.scatter(enter["Reviews"],enter["Category"],)
plt.xlabel("reviews")
plt.ylabel("category")
plt.show()

######################################### Collaborative filtering
#WE can see that supernatural movies are having less rating overall. Fantasy movies are having high rating overall.
#getting all the english stopwords using TfidfVectorizer
tfidf=TfidfVectorizer(stop_words="english")

#We have to fit this stopwords to category variable to remove stopwords from category
tfidf_matrix = tfidf.fit_transform(enter.Category)   #This makes all categories into matrix form
tfidf_matrix
tfidf_matrix.shape       #so totally there are 51 rows with 34 different categories

#cosine similarity
#We are checking the similarity of category matrix using cosine.
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim_matrix

#creating a mapping
#taking titles and giving them index
enter_index = pd.Series(enter.index, index = enter['Titles']).drop_duplicates()
enter_index

#checking that would give index values when we give the title
enter_id = enter_index["American President, The (1995)"]
enter_id

#writing a function to get recommendations
def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    enter_id = enter_index[Name]
    
    # Getting the pair wise similarity score
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[enter_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    enter_idx  =  [i[0] for i in cosine_scores_N]
    enter_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    enter_similar_show = pd.DataFrame(columns=["Titles", "Score"])
    enter_similar_show["Titles"] = enter.loc[enter_idx, "Titles"]
    enter_similar_show["Score"] = enter_scores
    enter_similar_show.reset_index(inplace = True)  
    print (enter_similar_show)
    
#Getting personalised recommendations for the movies
#when we give movie name, it suggests top 10 movies
get_recommendations("Heat (1995)", topN = 10)
















