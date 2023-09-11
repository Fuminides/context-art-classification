'''

Loads the mix dataset from all the dictionaries.
Computes labels for images in the iconclass images dataset.

Generates also som reports about the number of samples per class.
'''

import pickle
import json
import iconclass

import load_mix as lm

ICONCLASS_DATASET_PATH = '/home/javier/Documents/Iconclass Dataset/'
IMAGE_ANNOTATIONS_DICT = 'dict_images_annotations.pckl'
IMAGE_TERMS_DICT = 'dict_images_terms.pckl'
VALIDATED_ANNOTATION_TERMS_DICT = 'purged_term_annotation_dict.pckl'
FINAL_ANNOTATION_TERMS_DICT = 'final_term_annotation_dict.pckl'



def stablish_images():
    '''
    Generates the dictionary that maps each annotation with its set of images.

    :return: a dictionary where each element is an annotation and the elements are lists of strings
    containing the names of the corresponding images linked to each notation.
    '''

    images_data = json.load(open(ICONCLASS_DATASET_PATH + 'data.json', 'r'))
    images_dict = {}
    for file in os.listdir(ICONCLASS_DATASET_PATH):
        try:
            annotations = images_data[file]
            for annotation in annotations:
                if '(' not in annotation:  # Discard variants
                    try:
                        try:
                            txt_note = iconclass.get(iconclass.get(annotation)['p'][2])['txt']['en']
                        except IndexError:
                            txt_note = iconclass.get(iconclass.get(annotation)['p'][-1])['txt']['en']

                        try:
                            a = images_dict[txt_note]
                            a.append(file)
                            images_dict[txt_note] = a
                        except KeyError:
                            images_dict[txt_note] = [file]
                    except TypeError:
                        pass
        except KeyError:
            pass

    return images_dict


def dictionary_report():
    '''
    Gives some metrics about the term-iamges dictionary.
    Number of total terms.
    Number of terms with at least one example.
    List with term, number of samples tuple.

    :return: a list with the metrics as explained before.
    '''
    with open(IMAGE_TERMS_DICT, 'rb') as handle:
        dataset = pickle.load(handle)

    number_terms = len(dataset)
    samples_per_class = [[key, len(ter)] for key, ter in dataset.items()]
    good_samples_per_class = [[key, len(ter)] for key, ter in dataset.items() if len(ter) > 0]
    samples_per_class = sorted(samples_per_class, key=lambda a: a[1])

    return number_terms, good_samples_per_class, samples_per_class


def annotation_image2term_image_dict():
    '''
    Loads the annotation for each image dictionary and returns another dictionary where
    each term is the key and the elements are lists of strings of the correspondent images.

    :return: dictionary where each term is the key and the elements are lists of strings
    of the correspondent images.
    '''

    try:
        with open(IMAGE_TERMS_DICT, 'rb') as handle:
            return pickle.load(handle)

    except FileNotFoundError:
        pass

    ann_dict = pickle.load(open(FINAL_ANNOTATION_TERMS_DICT, 'rb'))
    terms = load_dicts().keys()
    concepts_iconography = ann_dict.keys()

    res = {}
    for term in terms:
        res[term] = []

        for concept in concepts_iconography:
            if concept in check_term_annotation(concept, term):
                res[term] += ann_dict[concept]

    with open(IMAGE_TERMS_DICT, 'wb') as handle:
        dataset = pickle.dump(res, handle)

    return res


def stablish_annotation_term_dict():
    # Establish manually corrected associations between terms nd annotations

    def purge_variations(ann_list, cut=3000):
        # Purge variations of the different annotations in iconclass. It also deletes the longest descriptions.
        import re
        ann = list(set([re.sub('\(.*\)', '', x).strip() for x in ann_list]))

        ann = sorted(ann, key=lambda a: len(a))[:cut]
        return ann

    def save_ter_ann_dict(new_dict):
        # Saves the dict in the pre-stablished path
        with open(VALIDATED_ANNOTATION_TERMS_DICT, 'wb') as f:
            pickle.dump(new_dict, f)

    annotations = list(pickle.load(open(IMAGE_ANNOTATIONS_DICT, 'rb')).keys())
    terms = list(pickle.load(open(lm.PURGED_TERMS_DICT, 'rb')).keys())
    annotations = purge_variations(annotations)

    try:
        with open(VALIDATED_ANNOTATION_TERMS_DICT, 'rb') as f:
            res = pickle.load(f)
    except FileNotFoundError:
        res = {}

    checked_terms = list(res.keys())

    i = 0
    total_terms = len(terms)
    term_counter = 0

    for term in terms:
        print('Term: ' + term)
        term_counter += 1
        print('Progreso: ' + str(term_counter) + ' out of ' + str(total_terms) + ' (' + str(term_counter / total_terms * 100) + '%)')

        if term not in checked_terms:
            succeses = 0
            for annotation in annotations:
                    i += 1

                    for annotation in annotations:
                        annotation = annotation.capitalize()
                        try:
                            proceed = not (annotation in res[term]) #If it is a new annotation for this term
                        except KeyError:
                            proceed = True

                        if proceed:
                            candidate = check_term_annotation(annotation, term)

                            '''if candidate and (annotation != term):
                                print('Term: ' + term)
                                print('Annotation: ' + annotation)

                                reinforcement = input('Good?')

                                if reinforcement == 'n':
                                    print('Next term!')
                                    break

                                good = reinforcement.startswith('y')'''
                            if candidate:
                                good = True
                            else:
                                good = False

                            if good:
                                succeses += 1
                                try:
                                    res[term].append(annotation)
                                except KeyError:
                                    res[term] = [annotation]

            print('Succeses: ' + str(succeses))


            if i % 10 == 0:
                save_ter_ann_dict(res)

    save_ter_ann_dict(res)
    return res

def show_images_term(term):
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    images_data = json.load(open(ICONCLASS_DATASET_PATH + 'data.json', 'r'))
    terms_images_dict = annotation_image2term_image_dict()

    for image in terms_images_dict[term]:
        img = mpimg.imread(ICONCLASS_DATASET_PATH + image)
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        for aa in images_data[image]:
            try:
                print(iconclass.get(aa)['txt']['en'])
            except TypeError:
                pass

        a = input('good?')
        plt.close()
        if a == 'e':
            return

def final_term_annotation():
    # Manually purges the candidate term/annotations dictionary
    with open(VALIDATED_ANNOTATION_TERMS_DICT, 'rb') as f:
        candidate_dict = pickle.load(f)

    final_dict = {}
    for term, annotations in candidate_dict.items():
        final_dict[term] = []

        for ann in annotations:
            if term != ann:
                print('Term: ' + term)
                print('Annotation: ' + ann)
                answer = input('good?')

                if answer.startswith('y'):
                    final_dict[term].append(ann)
                elif answer.startswith('n'): # Next term
                    break
            else:
                final_dict[term].append(ann)

            with open(FINAL_ANNOTATION_TERMS_DICT, 'wb') as f:
                print('Saving...')
                pickle.dump(final_dict, f)

    with open(FINAL_ANNOTATION_TERMS_DICT, 'wb') as f:
        pickle.dump(final_dict, f)


def check_term_annotation(annotation, term):
    # Returns true if we consider that the term is similar or equal to the annotated category.
    encontrado = False
    if len(term) == 1:
        if term in annotation.split(' '):
            encontrado = True
    else:
        if term in annotation:
            encontrado = True

    return encontrado


if __name__ == '__main__':
    #prueba = stablish_annotation_term_dict()
    #final_term_annotation()
    show_images_term('Christ')