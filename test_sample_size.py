# Evaluation of the size of scores sample

import numpy as np
from deepsig import bootstrap_power_analysis
from datetime import datetime

def get_scores_7_roles():
  models, precision, recall, f1 = [], [], [], []

  models.append('SingleSC-InCaseLaw') # rep-singlesc-InCaseLaw_2023-04-26-19h36min.txt
  precision.append([0.666496, 0.623050, 0.638246, 0.658660, 0.632378])
  recall.append([0.578162, 0.595206, 0.607495, 0.607375, 0.611875])
  f1.append([0.596474, 0.587752, 0.611066, 0.614648, 0.611059])

  models.append('SingleSC-RoBERTa') # rep-singlesc-RoBERTa_2023-04-26-19h38min.txt
  precision.append([0.602544, 0.601875, 0.586366, 0.595103, 0.580757])
  recall.append([0.585848, 0.585190, 0.569601, 0.569987, 0.578189])
  f1.append([0.585240, 0.585315, 0.568342, 0.572312, 0.574928])

  models.append('Cohan-InCaseLaw') # rep-Cohan_InCaseLaw_2023-05-02-17h26m42s.txt
  precision.append([0.654454, 0.686325, 0.706870, 0.681275, 0.659819])
  recall.append([0.635066, 0.647919, 0.627027, 0.616074, 0.644119])
  f1.append([0.622421, 0.647648, 0.652016, 0.627470, 0.637785])

  models.append('Cohan-RoBERTa') # rep-Cohan_RoBERTa_2023-05-02-17h27m42s.txt
  precision.append([0.699952, 0.700696, 0.712904, 0.687146, 0.711084])
  recall.append([0.601949, 0.612484, 0.603692, 0.615302, 0.615971])
  f1.append([0.629879, 0.639392, 0.625779, 0.639850, 0.647165])

  models.append('Cohan-Longformer') # 
  precision.append([0.720767, 0.678177, 0.699388, 0.707538, 0.712117])
  recall.append([0.651625, 0.635780, 0.660079, 0.654592, 0.643879])
  f1.append([0.673231, 0.651035, 0.671695, 0.668278, 0.666716])

  models.append('Mixup-InCaseLaw') # rep-mixup-InCaseLaw_2023-05-02-16h32min.txt
  precision.append([0.623156, 0.624863, 0.624285, 0.634015, 0.605722])
  recall.append([0.603832, 0.611683, 0.609120, 0.609434, 0.601673])
  f1.append([0.596707, 0.604771, 0.603974, 0.605986, 0.591493])

  models.append('Mixup-RoBERTa') # rep-mixup-RoBERTa_2023-05-02-12h51min.txt
  precision.append([0.588607, 0.587835, 0.560484, 0.576245, 0.575419])
  recall.append([0.584439, 0.581685, 0.573729, 0.571076, 0.573253])
  f1.append([0.58267, 0.578397, 0.561622, 0.567431, 0.570606])

  models.append('Mixup2-InCaseLaw') # rep-mixup2-InCaseLaw_2023-05-03-16h14m01s.txt
  precision.append([0.630066, 0.593929, 0.635539, 0.615454, 0.651481])
  recall.append([0.609143, 0.604425, 0.626980, 0.609594, 0.601307])
  f1.append([0.607601, 0.591196, 0.618036, 0.595636, 0.603050])

  models.append('Mixup2-RoBERTa') # rep-mixup2-RoBERTa_2023-05-03-16h13m45s.txt
  precision.append([0.588245, 0.591285, 0.581839, 0.586622, 0.583759])
  recall.append([0.585399, 0.595502, 0.573588, 0.593001, 0.573773])
  f1.append([0.579159, 0.586731, 0.570831, 0.577258, 0.572091])

  models.append('PE-S-InCaseLaw') # rep-pe-S-InCaseLaw_2023-05-05-14h02m28s.txt
  precision.append([0.619135, 0.611172, 0.612788, 0.613657, 0.606037])
  recall.append([0.598554, 0.614144, 0.595369, 0.605896, 0.606257])
  f1.append([0.595704, 0.605379, 0.593047, 0.596523, 0.596364])

  models.append('PE-S-RoBERTa') # rep-pe-S-RoBERTa_2023-05-05-14h02m30s.txt
  precision.append([0.553019, 0.586391, 0.576694, 0.580071, 0.567398])
  recall.append([0.539663, 0.569620, 0.559422, 0.567106, 0.570974])
  f1.append([0.543265, 0.572401, 0.565081, 0.567685, 0.566195])

  models.append('PE-C-InCaseLaw') # rep-pe-C-InCaseLaw_2023-05-05-14h06m58s.txt
  precision.append([0.637700, 0.617975, 0.619278, 0.644848, 0.640094])
  recall.append([0.627045, 0.616250, 0.607179, 0.626049, 0.619723])
  f1.append([0.618964, 0.604590, 0.602897, 0.623989, 0.617663])

  models.append('PE-C-RoBERTa') # rep-pe-C-RoBERTa_2023-05-05-14h07m05s.txt
  precision.append([0.587163, 0.614773, 0.551828, 0.564542, 0.582899])
  recall.append([0.586785, 0.600600, 0.556242, 0.573138, 0.578372])
  f1.append([0.577982, 0.599641, 0.549191, 0.560481, 0.572378])

  models.append('DFCSC-CLS-InCaseLaw') # rep-DFCSC-CLS_InCaseLaw_2023-05-04-13h56m32s.txt
  precision.append([0.668347, 0.700720, 0.669766, 0.720677, 0.659318])
  recall.append([0.650802, 0.658863, 0.630361, 0.642876, 0.642403])
  f1.append([0.646952, 0.666407, 0.643982, 0.663567, 0.639245])

  models.append('DFCSC-CLS-RoBERTa') # rep-DFCSC-CLS_RoBERTa_2023-05-04-13h52m35s.txt
  precision.append([0.754982, 0.719015, 0.725783, 0.737312, 0.737724])
  recall.append([0.676596, 0.665775, 0.694084, 0.680318, 0.675954])
  f1.append([0.706848, 0.684120, 0.702606, 0.701327, 0.695672])

  models.append('DFCSC-CLS-Longformer') # rep-DFCSC-CLS_Longformer_2023-05-04-14h41m37s.txt
  precision.append([0.787492, 0.742746, 0.775898, 0.753147, 0.784638])
  recall.append([0.688029, 0.696125, 0.697867, 0.680481, 0.703969])
  f1.append([0.722584, 0.711975, 0.726014, 0.707323, 0.734967])

  models.append('DFCSC-SEP-InCaseLaw') # rep-DFCSC-SEP_InCaseLaw_2023-05-09-10h59m31s.txt
  precision.append([0.693492, 0.658336, 0.697110, 0.693823, 0.697275])
  recall.append([0.662108, 0.653699, 0.652589, 0.635153, 0.668554])
  f1.append([0.661010, 0.651114, 0.662578, 0.653015, 0.666557])

  models.append('DFCSC-SEP-RoBERTa') # rep-DFCSC-SEP_RoBERTa_2023-05-09-10h59m41s.txt
  precision.append([0.739361, 0.751247, 0.742160, 0.721199, 0.726766])
  recall.append([0.684692, 0.683744, 0.693445, 0.670268, 0.675360])
  f1.append([0.706107, 0.708088, 0.710504, 0.689923, 0.694572])

  models.append('DFCSC-SEP-Longformer') # rep-DFCSC-SEP_Longformer_2023-05-09-10h59m41s.txt
  precision.append([0.790866, 0.768429, 0.765598, 0.762909, 0.776363])
  recall.append([0.701518, 0.699527, 0.706825, 0.704785, 0.689039])
  f1.append([0.730229, 0.724038, 0.730289, 0.723288, 0.721550])
  
  return models, precision, recall, f1

def get_scores_4_roles():
  models, precision, recall, f1 = [], [], [], []

  models.append('SingleSC-InCaseLaw') # rep-singlesc-InCaseLaw_2023-05-09-12h32m53s.txt
  precision.append([0.607578, 0.606670, 0.609534, 0.610767, 0.614338])
  recall.append([0.659139, 0.669460, 0.682907, 0.641407, 0.656738])
  f1.append([0.627019, 0.630515, 0.635151, 0.617589, 0.628652])

  models.append('SingleSC-RoBERTa') # rep-singlesc-RoBERTa_2023-05-09-16h15m54s.txt
  precision.append([0.607709, 0.606941, 0.613322, 0.610238, 0.602909])
  recall.append([0.635614, 0.646949, 0.625476, 0.643613, 0.619878])
  f1.append([0.618076, 0.621857, 0.616595, 0.624315, 0.604194])

  models.append('Mixup-InCaseLaw') # rep-mixup-InCaseLaw_2023-05-10-07h49min.txt
  precision.append([0.608290, 0.603896, 0.614796, 0.600443, 0.607498])
  recall.append([0.668176, 0.663238, 0.667873, 0.657826, 0.656222])
  f1.append([0.631599, 0.627935, 0.636509, 0.623736, 0.627542])

  models.append('Mixup-RoBERTa') # rep-mixup-RoBERTa_2023-05-09-20h45min.txt
  precision.append([0.601268, 0.589125, 0.607986, 0.599014, 0.608088])
  recall.append([0.618254, 0.632461, 0.628675, 0.636784, 0.627519])
  f1.append([0.609105, 0.605977, 0.616340, 0.615063, 0.616950])

  models.append('Mixup2-InCaseLaw') # rep-mixup2-InCaseLaw_2023-05-09-20h48m37s.txt
  precision.append([0.603651, 0.617353, 0.617231, 0.612939, 0.608055])
  recall.append([0.663769, 0.669212, 0.671943, 0.667487, 0.655847])
  f1.append([0.626331, 0.638211, 0.638530, 0.633378, 0.625712])

  models.append('Mixup2-RoBERTa') # rep-mixup2-RoBERTa_2023-05-09-20h48m46s.txt
  precision.append([0.598562, 0.614282, 0.597493, 0.586531, 0.593967])
  recall.append([0.632729, 0.653608, 0.634763, 0.639884, 0.639257])
  f1.append([0.613461, 0.630332, 0.612929, 0.608218, 0.612780])

  models.append('PE-S-InCaseLaw') # rep-pe-S-InCaseLaw_2023-05-09-12h33m23s.txt
  precision.append([0.626847, 0.615584, 0.627770, 0.597641, 0.620444])
  recall.append([0.666219, 0.650833, 0.659821, 0.647789, 0.654544])
  f1.append([0.640741, 0.629386, 0.638751, 0.614264, 0.633903])

  models.append('PE-S-RoBERTa') # rep-pe-S-RoBERTa_2023-05-09-16h16m40s.txt
  precision.append([0.624745, 0.621288, 0.615196, 0.592504, 0.619538])
  recall.append([0.661754, 0.631590, 0.639963, 0.638632, 0.636102])
  f1.append([0.639982, 0.625129, 0.623726, 0.608659, 0.624303])

  models.append('PE-C-InCaseLaw') # rep-pe-C-InCaseLaw_2023-05-09-12h34m13s.txt
  precision.append([0.633993, 0.593802, 0.608673, 0.627178, 0.638192])
  recall.append([0.669336, 0.645320, 0.655616, 0.663733, 0.692449])
  f1.append([0.646868, 0.609477, 0.626605, 0.641143, 0.658333])

  models.append('PE-C-RoBERTa') # rep-pe-C-RoBERTa_2023-05-09-16h16m52s.txt
  precision.append([0.615617, 0.612486, 0.649709, 0.590801, 0.607026])
  recall.append([0.653562, 0.647093, 0.657568, 0.619459, 0.613029])
  f1.append([0.628850, 0.623648, 0.651139, 0.600856, 0.607864])

  models.append('Cohan-InCaseLaw') # rep-Cohan_InCaseLaw_2023-05-09-22h55m06s.txt
  precision.append([0.689467, 0.696641, 0.650893, 0.665068, 0.682871])
  recall.append([0.696484, 0.659998, 0.677705, 0.664699, 0.655428])
  f1.append([0.692356, 0.676338, 0.660846, 0.664188, 0.667880])

  models.append('Cohan-RoBERTa') # rep-Cohan_RoBERTa_2023-05-09-23h29m16s.txt
  precision.append([0.660138, 0.690244, 0.678006, 0.652304, 0.663673])
  recall.append([0.650915, 0.608564, 0.663337, 0.617682, 0.652915])
  f1.append([0.654419, 0.633174, 0.669563, 0.628730, 0.654598])

  models.append('Cohan-Longformer') # rep-Cohan_Longformer_2023-05-09-23h41m40s.txt
  precision.append([0.678618, 0.691750, 0.671975, 0.642334, 0.671450])
  recall.append([0.648360, 0.635654, 0.634373, 0.639322, 0.646338])
  f1.append([0.662085, 0.657262, 0.649332, 0.639470, 0.657608])

  models.append('DFCSC-CLS-InCaseLaw') # rep-DFCSC-CLS_InCaseLaw_2023-05-10-07h12m35s.txt
  precision.append([0.654944, 0.625410, 0.668478, 0.633049, 0.655702])
  recall.append([0.644460, 0.662850, 0.647263, 0.639584, 0.668652])
  f1.append([0.648763, 0.638407, 0.656793, 0.634681, 0.661043])

  models.append('DFCSC-CLS-RoBERTa') # rep-DFCSC-CLS_RoBERTa_2023-05-10-07h12m53s.txt
  precision.append([0.698986, 0.704526, 0.685595, 0.691256, 0.689918])
  recall.append([0.677277, 0.706161, 0.699280, 0.651226, 0.683135])
  f1.append([0.685035, 0.704886, 0.689684, 0.667206, 0.685323])

  models.append('DFCSC-CLS-Longformer') # rep-DFCSC-CLS_Longformer_2023-05-10-07h13m16s.txt
  precision.append([0.732052, 0.714100, 0.724818, 0.715393, 0.686494])
  recall.append([0.730059, 0.678948, 0.706081, 0.728211, 0.674010])
  f1.append([0.728509, 0.693937, 0.713957, 0.719376, 0.679353])

  models.append('DFCSC-SEP-InCaseLaw') # rep-DFCSC-SEP_InCaseLaw_2023-05-10-07h15m10s.txt
  precision.append([0.688666, 0.649366, 0.645186, 0.669574, 0.638603])
  recall.append([0.662243, 0.667662, 0.622190, 0.686087, 0.661596])
  f1.append([0.674053, 0.656175, 0.632651, 0.674267, 0.648341])

  models.append('DFCSC-SEP-RoBERTa') # rep-DFCSC-SEP_RoBERTa_2023-05-10-07h47m08s.txt
  precision.append([0.683767, 0.694597, 0.713471, 0.704070, 0.707609])
  recall.append([0.671728, 0.697525, 0.708944, 0.683974, 0.704685])
  f1.append([0.677448, 0.694904, 0.710989, 0.692819, 0.703286])

  models.append('DFCSC-SEP-Longformer') # rep-DFCSC-SEP_Longformer_2023-05-10-07h46m57s.txt
  precision.append([0.713992, 0.733078, 0.704516, 0.705189, 0.716620])
  recall.append([0.729323, 0.696979, 0.721106, 0.698671, 0.720839])
  f1.append([0.721341, 0.712153, 0.709626, 0.701567, 0.716092])
  
  return models, precision, recall, f1

'''
models.append('')
precision.append([])
recall.append([])
f1.append([])
'''
def evaluate_sample_sizes(models, precision, recall, f1):
  out = ''
  for i in range(len(models)):
    out += models[i] + '\n'
    out += f'  Precision power: {bootstrap_power_analysis(np.array(precision[i]), show_progress=False)}\n'
    out += f'  Recall power:    {bootstrap_power_analysis(np.array(recall[i]), show_progress=False)}\n'
    out += f'  F1 power:        {bootstrap_power_analysis(np.array(f1[i]), show_progress=False)}\n'
  return out

def main():
  report = ''
  
  report += '======== 7-roles Dataset ========\n'
  report += evaluate_sample_sizes(*get_scores_7_roles())

  report += '\n======== 4-roles Dataset ========\n'
  report += evaluate_sample_sizes(*get_scores_4_roles())
  
  # saving report
  time_tag = f'{datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")}'
  with open(f'./rep-bootstrap-power-{time_tag}.txt', 'w') as f:
    f.write(report)

if __name__ == "__main__":
    main()