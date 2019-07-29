from pygments.lexer import RegexLexer, bygroups,words
from pygments.token import *

class CustomLexer(RegexLexer):
    name = 'Example Lexer for test desc'

    tokens = {
        'root': [
            (words(("group","suite","class","folder","ParamList")), Keyword),
            (words(("Pattern","Output","Params","oldID")), Keyword.Type),
            (words(("Summary","Names","Formula","Functions")), Name.Function),
            (r'\"[^\"]*\"',String),
            (r'[a-zA-Z_][a-zA-Z_0-9\.]*',Text),
            (r'[-+]?[0-9]+',Number),
            (r'([=:\[\]]|->)',Operator),
            (r'[\s\t]+',Text),
            (r'[,{}]',Text),
            (r'//.*?$', Comment.Singleline),
        ]
    }