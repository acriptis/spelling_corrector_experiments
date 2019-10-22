from ELMO_inference import ELMOLM

for i in range(2):
    elmo = ELMOLM(model_dir='~/.deeppavlov/downloads/embeddings/rus_vectors_cpkt/')
    print(elmo.estimate_likelihood('я гуляла по траве'))
    print(elmo.estimate_likelihood('я гуляла по облакам'))

    print(elmo.estimate_likelihood_batch(['я гуляла по траве', 'я гуляла по ываыа']))
    print()
