from .engine import Classifier

def main():
    classifier = Classifier()
    classifier.run(processed_data=False, vanilla_classifier=False, custom_features=False)

main()