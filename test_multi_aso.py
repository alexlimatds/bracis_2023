import numpy as np
import deepsig
import test_sample_size
from datetime import datetime

def run_test(ds_name, models, scores):
  seed = 1234
  np.random.seed(seed)
  use_bonferroni = True
  confidence_level = 0.95

  #model_to_scores = {m : s for m, s in zip(models[0:3], scores[0:3])} # for test purposes
  model_to_scores = {m : s for m, s in zip(models, scores)}
  eps_min = deepsig.multi_aso(
    model_to_scores, 
    confidence_level=confidence_level, 
    return_df=True, 
    num_jobs=-1, 
    use_bonferroni=use_bonferroni, 
    seed=seed
  )

  report = f'\n======= {ds_name} =======\nBonferroni correction: {use_bonferroni}\nConfidence level: {confidence_level}\n' + eps_min.to_string() + '\n'
  return report

def main():
  report = ''
  
  # 7-labels dataset
  models, _, _, f1_scores = test_sample_size.get_scores_7_roles()
  report += run_test('7-roles dataset', models, f1_scores)
  
  # 4-labels dataset
  models, _, _, f1_scores = test_sample_size.get_scores_4_roles()
  report += run_test('4-roles dataset', models, f1_scores)
  
  # saving report
  time_tag = f'{datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")}'
  with open(f'./rep-multi_aso-{time_tag}.txt', 'w') as f:
    f.write(report)

if __name__ == "__main__":
  main()