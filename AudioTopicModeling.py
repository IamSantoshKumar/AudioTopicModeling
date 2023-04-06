import whisper
import argparse
from pprint import pprint
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

"""
Audio to topic modeling is the process of extracting meaningful topics or themes from audio data. 
This can be done using a variety of techniques, such as speech recognition, natural language processing,
and machine learning algorithms.
"""

class AudioTopicModeling:
    
    def __init__(self, model_name=None):
        self.model = whisper.load_model(model_name)
        self.vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1,3))
        self.topic_model = BERTopic(vectorizer_model=self.vectorizer_model)
        
    def __call__(self, audio_file=None):
        result = self.model.transcribe(audio_file) 
        topics, probs = self.topic_model.fit_transform(str(result["text"]).split(","))
        return self.topic_model.get_topic(0) 
    
    def visualize_topic(self, n_topics):
        return self.topic_model.visualize_barchart(top_n_topics=n_topics)
         
    def visualize_hierarchy(self, n_topics):
        return self.topic_model.visualize_hierarchy(top_n_topics=n_topics)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--modelname", type=str)
    parser.add_argument("--audiofile", type=str)

    args = parser.parse_args()
    
    generate_topic = AudioTopicModeling(model_name=args.modelname)
    
    pprint(generate_topic(args.audiofile))

