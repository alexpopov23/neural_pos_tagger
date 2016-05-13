import string

def get_suffix (word):

    title, isAllCaps, isLower, isDigit = False, False, False, False
    if word.istitle():
        title = True
    if word.isupper():
        isAllCaps = True
    if word.islower():
        isLower = True
    if word.isdigit():
        isDigit = True

    if isDigit:
        word = len(word)*"0"
        return word
    if len(word) > 3:
        word = word[-3:]
        word = "_" + word
    if len(word) > 1 and isAllCaps:
        word = "$" + word
    if title:
        word = "@" + word
    if not (title or isLower or isAllCaps or word in string.punctuation):
        word = "#" + word
    return word.lower()

