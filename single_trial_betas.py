import sys
import json

#Converts BIDS formatted jsons into jsons with single trial beta outputs
#Takes original json path, target json directory path, target condition, and the number of total trials
#Generates json for each trial of each condition with single trial beta output
def single_trial_beta_model(json_orig_path, json_new_dir, condition_name, num_trials):
    for trial_n in range(num_trials):
        with open(json_orig_path, 'r') as h:
            jdat = json.load(h)

        #Replaces target condition with target trial and other trials regressors
        jdat["Steps"][0]['Model']['X'].remove(condition_name)
        jdat["Steps"][0]['Model']['X'] = [f"trial-{trial_n:03d}_{condition_name}",f"other_trials_{condition_name}"] + jdat["Steps"][0]['Model']['X']

        #Removes all model levels other than 1st level, contrasts, and existing dummy contrasts
        jdat['Steps'] = [jdat['Steps'][0]]
        jdat['Steps'][0].pop('Contrasts', None)
        jdat['Steps'][0].pop('DummyContrasts', None)

        #Adds target trial regressors to convolve list
        to_convolve = [x for x in jdat["Steps"][0]['Transformations'] if x["Name"] == "Convolve"][0]["Input"]
        to_convolve.remove(condition_name)
        #to_convolve.append(f"trial-{trial_n:03d}_{condition_name}")
        #to_convolve.append(f"other_trials_{condition_name}")
        to_convolve = [f"trial-{trial_n:03d}_{condition_name}",f"other_trials_{condition_name}"] + to_convolve
        [x for x in jdat["Steps"][0]['Transformations'] if x["Name"] == "Convolve"][0]["Input"] = to_convolve

        #Adds target trial regressor to dummy contrast
        jdat['DummyContrasts'] = {
            "Conditions": [f"trial-{trial_n:03d}_{condition_name}"],
            "Type":"t"
        }
        #generates output json
        json_new_path = f'{json_new_dir}stb_trial-{trial_n:03d}_{condition_name}.json'
        with open(json_new_path, 'w') as h:
            json.dump(jdat, h, indent=2)



if __name__ == "__main__": 
    json_orig_path = sys.argv[1]
    json_new_dir = sys.argv[2]
    condition_name = sys.argv[3]
    num_trials = sys.argv[4]
    
    single_trial_beta_model(json_orig_path, json_new_dir, condition_name, int(num_trials))