import os
import streamlit as st

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow informational and warning logs

# Remove dynamic installation of beautifulsoup4
from bs4 import BeautifulSoup

from transformers import pipeline
#plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """Cristiano Ronaldo and Lionel Messi will go head-to-head once more in the race to be this season's top scorer in the Champions League – although Luiz Adriano threatens to spoil the party. 
            Both Barcelona and Real Madrid booked their spots in the semi-finals this week with victories over Paris Saint-Germain and Atletico Madrid respectively. 
            The planet's best footballers have scored eight times in Europe this season. But Shakhtar Donetsk’s Adriano, courted by Arsenal and Liverpool, has netted on nine occasions this term. 
            Cristiano Ronaldo, in action against Atletico Madrid on Wednesday evening, has scored eight goals in Europe . 
            Lionel Messi also has eight goals in the Champions League this term; one fewer than Luiz Adriano . 
            Ronaldo and Messi will both play at least two more times after Real Madrid and Barcelona reached the last four . 
            Adriano, who moved to Donetsk in 2007, scored five against BATE Borsiov in the group stages. 
            His performance that night made history, with the 27-year-old becoming only the second player to score five times in a Champions League game. 
            The other was Messi for Barcelona against Bayer Leverkusen in 2012. He also scored the third quickest hat-trick in the competition's history (12 minutes) as the Ukrainian side, knocked out by Bayern Munich in the round of 16, racked up the biggest-ever half-time lead (6-0) in Europe's premier tournament. 
            ‘I am in a good moment of my career and we'll do what will be best for me and for the club,’ said Adriano last month when quizzed over his future. 
            Adriano, who netted five times against BATE Borisov in the group, has scored more goals than any other player in the Champions League... he is out of contract in December and could move to the Premier League . 
            ‘With my contract set to expire and many good performances, it'll be difficult to stay in Ukraine.’ Arsenal have sent scouts to watch Adriano in recent months, while Liverpool are also keen on the Brazilian. 
            His contract with Shakhtar Donetsk runs out at the end of the year. Ronaldo and Messi however, remain in pole-position to top the scoring charts with Barcelona and Real Madrid both in the hat for the two-legged semi-finals to be played next month. 
            Of the teams still in the pot, Neymar and Luis Suarez of Barcelona, Real Madrid's Karim Benzema and former Manchester United and City striker Carlos Tevez, now plying his trade for Juventus, each have six goals. 
            The draw for the last four will take place on Friday."""
st.title("Article Summarization")
st.markdown("This app summarizes articles using the BART model. Paste your article in the text area below and click 'Summarize' to get the summary.")
st.write("Below is a sample article for demonstration purposes. You can replace it with your own article.")

st.text_area("Paste your article here:", height=300, value=ARTICLE)
summary_button = st.button("Summarize")

if summary_button:
    # Display the summary   
    st.write("Summary:")
    summary = summarizer(ARTICLE, max_length=130, min_length=75, do_sample=False)
    st.write(summary[0]['summary_text'])