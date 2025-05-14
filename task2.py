import pandas as pd
import numpy as np
from scipy.stats import norm

def main():
    #getting the data ready
    file_path = 'detection_data.csv'
    data = pd.read_csv(file_path)
    data_2 = pd.read_csv('detection_data_extra.csv')
    detect_data = data[data['Detection'] == 'Detect']
    no_detect_data = data[data['Detection'] == 'No Detect']

    #calculating mean and variance for detected and not detected points 
    mean_amplitude_detect = detect_data['Amplitude'].mean()
    var_amplitude_detect = detect_data['Amplitude'].var()
    mean_distance_detect = detect_data['Distance'].mean()
    var_distance_detect = detect_data['Distance'].var()

    mean_amplitude_no_detect = no_detect_data['Amplitude'].mean()
    var_amplitude_no_detect = no_detect_data['Amplitude'].var()
    mean_distance_no_detect = no_detect_data['Distance'].mean()
    var_distance_no_detect = no_detect_data['Distance'].var()

    
    #function which abuses bayes rule to deduce if a point should be guessed as detect or not detect
    def calculate_probability(amplitude, distance):
        #find the probabilites assuming normal distribution for amplitude and distance when detect
        p_a_detect = norm.pdf(amplitude, loc=mean_amplitude_detect, scale=np.sqrt(var_amplitude_detect))
        p_d_detect = norm.pdf(distance, loc=mean_distance_detect, scale=np.sqrt(var_distance_detect))
        p_detect_given_a_d = p_a_detect * p_d_detect #  p_detect_given_a_d α p_a_detect * p_d_detect in reality

        #find the probabilites assuming normal distribution for amplitude and distance when no detect
        p_a_no_detect = norm.pdf(amplitude, loc=mean_amplitude_no_detect, scale=np.sqrt(var_amplitude_no_detect))
        p_d_no_detect = norm.pdf(distance, loc=mean_distance_no_detect, scale=np.sqrt(var_distance_no_detect))
        p_no_detect_given_a_d = p_a_no_detect * p_d_no_detect # p_a_no_detect * p_d_no_detect α p_no_detect_given_a_d  in reality


        return {
            'P(Detect | a, d)': p_detect_given_a_d,
            'P(No Detect | a, d)': p_no_detect_given_a_d,
            'Prediction': 'Detect' if p_detect_given_a_d > p_no_detect_given_a_d else 'No Detect'
        }

    def evaluate_model(data):
        correct_predictions = 0
        total_predictions = len(data)

        for _, row in data.iterrows():
            amplitude = row['Amplitude']
            distance = row['Distance']
            actual = row['Detection']

            prediction = calculate_probability(amplitude, distance)['Prediction']

            if prediction == actual:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    evaluate_model(data_2)    

if __name__ == "__main__":
    main()
