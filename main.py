import sys
sys.path.append('..')
from common.util import preprocess, create_contexts_target, convert_one_hot

def main():
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    contexts, target = create_contexts_target(corpus, window_size=1)

    vocab_size = len(word_to_id)
    target = convert_one_hot(target, vocab_size)

    contexts = convert_one_hot(contexts, vocab_size)

    print(vocab_size)


# if __name__ == 'main':
main()