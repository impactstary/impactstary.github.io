/* Prism.js - Minimal Build
 * A lightweight, self-contained syntax highlighter
 * Supports: markup, css, clike, javascript, python, c, cpp, bash, json, markdown, sql
 * Based on Prism.js (prismjs.com) - MIT License
 */
(function (global) {
  'use strict';

  var Prism = (function () {

    var Prism = {
      util: {},
      languages: {},
      plugins: {},
      highlightAll: highlightAll,
      highlightAllUnder: highlightAllUnder,
      highlight: highlight,
      highlightElement: highlightElement,
      fileType: fileType,
      Token: Token
    };

    // ========== Utilities ==========

    var util = Prism.util;

    util.type = function (o) {
      return Object.prototype.toString.call(o).slice(8, -1);
    };

    util.objId = function (obj) {
      if (!obj['__id']) {
        Object.defineProperty(obj, '__id', { value: Math.random() });
      }
      return obj['__id'];
    };

    util.clone = function (o, visited) {
      var type = util.type(o);
      visited = visited || {};

      switch (type) {
        case 'Object':
          var id = util.objId(o);
          if (visited[id]) return visited[id];
          var result = {};
          visited[id] = result;
          for (var key in o) {
            if (o.hasOwnProperty(key)) {
              result[key] = util.clone(o[key], visited);
            }
          }
          return result;

        case 'Array':
          var id = util.objId(o);
          if (visited[id]) return visited[id];
          var result = [];
          visited[id] = result;
          for (var i = 0, l = o.length; i < l; i++) {
            result.push(util.clone(o[i], visited));
          }
          return result;

        default:
          return o;
      }
    };

    util.encode = function encode(tokens) {
      if (tokens instanceof Token) {
        return new Token(tokens.type, encode(tokens.content), tokens.alias);
      } else if (util.type(tokens) === 'Array') {
        return tokens.map(encode);
      } else {
        return tokens
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/\u00a0/g, ' ');
      }
    };

    util.LanguageDefinition = function LanguageDefinition() {};

    util.LanguageDefinition.prototype = new Function();

    util.extend = function (id, redef) {
      var lang = util.clone(Prism.languages[id]);
      for (var key in redef) {
        lang[key] = redef[key];
      }
      return lang;
    };

    util.insertBefore = function (inside, before, insert, root) {
      root = root || Prism.languages;
      var grammar = root[inside];
      var ret = {};

      for (var token in grammar) {
        if (grammar.hasOwnProperty(token)) {
          if (token == before) {
            for (var newToken in insert) {
              if (insert.hasOwnProperty(newToken)) {
                ret[newToken] = insert[newToken];
              }
            }
          }
          ret[token] = grammar[token];
        }
      }

      // Update references
      var old = root[inside];
      root[inside] = ret;
      return ret;
    };

    util.DFS = function DFS(o, callback, type, visited) {
      visited = visited || {};
      var id = util.objId(o);
      if (visited[id]) return;
      visited[id] = true;

      callback(o, type);

      if (util.type(o) === 'Object') {
        for (var i in o) {
          if (o.hasOwnProperty(i)) {
            DFS(o[i], callback, i, visited);
          }
        }
      } else if (util.type(o) === 'Array') {
        o.forEach(function (i, idx) { DFS(i, callback, idx, visited); });
      }
    };

    // ========== Token Class ==========

    function Token(type, content, alias, matchedStr, greedy) {
      this.type = type;
      this.content = content;
      this.alias = alias;
      this.length = (matchedStr || '').length | 0;
      this.greedy = !!greedy;
    }

    Token.stringify = function stringify(o, language) {
      if (typeof o == 'string') return o;
      if (util.type(o) == 'Array') {
        return o.map(function (element) {
          return stringify(element, language);
        }).join('');
      }

      var env = {
        type: o.type,
        content: stringify(o.content, language),
        tag: 'span',
        classes: ['token', o.type],
        attributes: {},
        language: language
      };

      var aliases = o.alias;
      if (aliases) {
        if (util.type(aliases) == 'Array') {
          env.classes.push.apply(env.classes, aliases);
        } else {
          env.classes.push(aliases);
        }
      }

      var attributes = '';
      for (var name in env.attributes) {
        attributes += ' ' + name + '="' + (env.attributes[name] || '').replace(/"/g, '&quot;') + '"';
      }

      return '<' + env.tag + ' class="' + env.classes.join(' ') + '"' + attributes + '>' + env.content + '</' + env.tag + '>';
    };

    // ========== Core Highlighting ==========

    function matchGrammar(text, tokenList, grammar, startNode, startPos, rematch) {
      for (var token in grammar) {
        if (!grammar.hasOwnProperty(token) || !grammar[token]) continue;

        var patterns = grammar[token];
        patterns = (util.type(patterns) === 'Array') ? patterns : [patterns];

        for (var j = 0; j < patterns.length; ++j) {
          var pattern = patterns[j],
            inside = pattern.inside,
            lookbehind = !!pattern.lookbehind,
            greedy = !!pattern.greedy,
            lookbehindLength = 0,
            alias = pattern.alias;

          if (greedy && !pattern.pattern.global) {
            var flags = pattern.pattern.toString().match(/[imuy]*$/)[0];
            pattern.pattern = RegExp(pattern.pattern.source, flags + 'g');
          }

          pattern = pattern.pattern || pattern;

          for (var i = startPos, pos = 0, len = tokenList.length; pos < len; i += tokenList[pos].length, ++pos) {
            var token = tokenList[pos];
            if (token.length == null) continue;
            if (i >= startNode.length) break;

            var deleteCount = 0;
            var matchedToken, matchArr = null;
            var k = pos;

            if (token.type == 'text' && !rematch) {
              pattern.lastIndex = i;
              matchArr = pattern.exec(startNode);
              if (matchArr) {
                var matchPos = matchArr.index;
                if (lookbehind && matchArr[1]) {
                  lookbehindLength = matchArr[1].length;
                }
                if (matchPos >= i && matchPos < i + token.length) {
                  var from = matchPos - i;
                  var to = from + matchArr[0].length;
                  var before = token.content.slice(0, from);
                  var match = token.content.slice(from, to);
                  var after = token.content.slice(to);

                  var newTokens = [];
                  if (before) newTokens.push(new Token('text', before, null, before));
                  matchedToken = new Token(token, inside ? tokenize(match, inside) : match, alias, match, greedy);
                  newTokens.push(matchedToken);
                  if (after) newTokens.push(new Token('text', after, null, after));

                  deleteCount = 1;
                  for (var z = pos + 1; z < tokenList.length && tokenList[z].greedy && tokenList[z].type == 'text'; z++) {
                    pattern.lastIndex = i + token.length;
                    var m = pattern.exec(startNode);
                    if (m && m.index == i + token.length) {
                      token = tokenList[z];
                      matchedToken = new Token(token.type, inside ? tokenize(token.content, inside) : token.content, alias, token.content, greedy);
                      newTokens.push(matchedToken);
                      deleteCount++;
                    } else {
                      break;
                    }
                  }

                  var args = [pos, deleteCount].concat(newTokens);
                  tokenList.splice.apply(tokenList, args);
                  if (rematch) {
                    matchGrammar(text, tokenList, grammar, startNode, i, false);
                  }
                  return;
                }
              }
            } else if (token.greedy && token.type != 'text') {
              for (; pos < len && tokenList[pos].type != 'text'; ++pos) {
                i += tokenList[pos].length;
              }
              --pos;
            }
          }
        }
      }
    }

    function tokenize(text, grammar) {
      var rest = grammar.rest;
      if (rest) {
        for (var token in rest) {
          grammar[token] = rest[token];
        }
        delete grammar.rest;
      }

      var tokenList = [new Token('text', text, null, text)];
      matchGrammar(text, tokenList, grammar, text, 0, false);

      return tokenList;
    }

    function highlight(text, grammar, language) {
      var env = {
        code: text,
        grammar: grammar,
        language: language
      };

      if (!env.grammar) {
        return util.encode(env.code);
      }

      env.tokens = tokenize(env.code, env.grammar);

      return Token.stringify(util.encode(env.tokens), env.language);
    }

    function highlightElement(element, async, callback) {
      var language = element.getAttribute('data-language') || getLanguage(element);
      var grammar = Prism.languages[language];

      element.setAttribute('data-language', language);

      var code = element.textContent;
      var env = {
        element: element,
        language: language,
        grammar: grammar,
        code: code
      };

      if (!env.grammar) {
        env.highlightedCode = env.code;
      } else {
        env.highlightedCode = Prism.highlight(env.code, env.grammar, env.language);
      }

      element.innerHTML = env.highlightedCode;
      element.classList.add('language-' + language);

      if (callback) callback(element);
    }

    function getLanguage(element) {
      while (element && element.classList) {
        var list = element.classList;
        for (var i = 0; i < list.length; i++) {
          var cls = list[i];
          if (cls.indexOf('language-') === 0) return cls.slice('language-'.length);
          if (cls.indexOf('lang-') === 0) return cls.slice('lang-'.length);
        }
        element = element.parentElement;
      }
      return 'none';
    }

    function fileType(type) {
      // Simple mapping
      return type;
    }

    function highlightAll(async, callback) {
      var elements = document.querySelectorAll('code[class*="language-"], [class*="language-"] code, code[class*="lang-"], [class*="lang-"] code');
      for (var i = 0; i < elements.length; i++) {
        highlightElement(elements[i], async, callback);
      }
    }

    // ========== Language: Markup (HTML/XML) ==========

    Prism.languages.markup = {
      'comment': /<!--[\s\S]*?-->/,
      'prolog': /<\?[\s\S]+?\?>/,
      'doctype': /<!DOCTYPE[\s\S]+?>/i,
      'cdata': /<!\[CDATA\[[\s\S]*?]]>/i,
      'tag': {
        pattern: /<\/?(?!\d)[^\s>\/=$<%]+(?:\s(?:\s*[^\s>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+(?=[\s>]))|(?=[\s/>])))+)?\s*\/?>/i,
        greedy: true,
        inside: {
          'tag': {
            pattern: /^<\/?[^\s>\/]+/i,
            inside: {
              'punctuation': /^<\/?/,
              'namespace': /^[^\s>\/:]+:/
            }
          },
          'attr-value': {
            pattern: /=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+)/i,
            inside: {
              'punctuation': [
                /^=/,
                {
                  pattern: /^(\s*)["']|["']$/,
                  lookbehind: true
                }
              ]
            }
          },
          'punctuation': /\/?>/,
          'attr-name': {
            pattern: /[^\s>\/]+/,
            inside: {
              'namespace': /^[^\s>\/:]+:/
            }
          }
        }
      },
      'entity': /&#?[\da-z]{1,8};/i
    };

    Prism.languages.markup.tag.inside['attr-value'].inside.entity = Prism.languages.markup.entity;

    Prism.languages.html = Prism.languages.markup;
    Prism.languages.xml = Prism.languages.markup;
    Prism.languages.mathml = Prism.languages.markup;
    Prism.languages.svg = Prism.languages.markup;

    // ========== Language: CSS ==========

    Prism.languages.css = {
      'comment': /\/\*[\s\S]*?\*\//,
      'atrule': {
        pattern: /@[\w-](?:[^;{]|url\([^)]*\))*[;{]/i,
        inside: {
          'keyword': /^@[\w-]+/,
          'punctuation': /[;{]$/
        }
      },
      'url': /url\((?:["'](?:\\.|[^"\\\n])*["']|(?:\\.|[^\s"')\\])*)\)/i,
      'selector': {
        pattern: /[^{}\s][^{};]*?(?=\s*\{)/,
        inside: {
          'attribute': /\[[^\]]+\]/,
          'pseudo-class': /:[\w-]+(?:\([^)]*\))?/i,
          'pseudo-element': /::[\w-]+/i,
          'id': /#[\w-]+/,
          'class': /\.[\w-]+/,
          'combinator': /[>+~]/,
          'namespace': /[\w-]*\|/
        }
      },
      'string': {
        pattern: /"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'/,
        greedy: true
      },
      'property': /[-_a-z\xA0-\uFFFF][-\w\xA0-\uFFFF]*(?=\s*:)/i,
      'important': /!important\b/i,
      'function': /[-a-z0-9]+(?=\()/i,
      'punctuation': /[{}();:,]/,
      'number': /-?(?:\d*\.?\d+)(?:e[+-]?\d+)?(?:%|ch|cm|em|ex|fr|in|mm|pc|pt|px|rem|vh|vmax|vmin|vw|s|ms|deg|rad|turn)?/i,
      'color': /#(?:[0-9a-f]{3}){1,2}\b/i
    };

    // ========== Language: CLike (base for C/C++/Java/etc) ==========

    Prism.languages.clike = {
      'comment': [
        {
          pattern: /(^|[^\\])\/\*[\s\S]*?(?:\*\/|$)/,
          lookbehind: true
        },
        {
          pattern: /(^|[^\\:])\/\/.*/,
          lookbehind: true,
          greedy: true
        }
      ],
      'string': {
        pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
        greedy: true
      },
      'class-name': {
        pattern: /((?:\b(?:class|interface|extends|implements|trait|instanceof|new)\s+)|(?:catch\s+\())[\w.\\]+/i,
        lookbehind: true,
        inside: {
          punctuation: /[.\\]/
        }
      },
      'keyword': /\b(?:if|else|while|do|for|return|in|instanceof|function|new|try|throw|catch|finally|break|continue|switch|case|default|void|delete|typeof|instanceof|var|let|const|class|extends|import|export|from|async|await|yield|public|private|protected|static|final|abstract|readonly|enum|struct|union|typedef|template|typename|namespace|using|explicit|friend|inline|virtual|operator|goto|volatile|register|auto|extern|this|super|true|false|null|undefined|NaN|Infinity)\b/,
      'boolean': /\b(?:true|false)\b/,
      'function': /[-a-zA-Z_$][\w$]*(?=\s*\()/,
      'number': /\b0x[\da-f]+\b|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:e[+-]?\d+)?[fFlL]?[uU]?\b/i,
      'operator': /--?|\+\+?|!=?=?|<=?|>=?|==?=?|&&?|\|\|?|\?|\*|\/|~|\^|%/,
      'punctuation': /[{}[\];(),.:]/
    };

    // ========== Language: JavaScript ==========

    Prism.languages.javascript = Prism.languages.extend('clike', {
      'class-name': [
        Prism.languages.clike['class-name'],
        {
          pattern: /(^|[^$\w\xA0-\uFFFF])[_$A-Z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\s+(?:extends|implements|instanceof)\s)/,
          lookbehind: true
        }
      ],
      'keyword': /\b(?:as|async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|export|extends|finally|for|from|function|get|if|import|in|instanceof|let|new|null|of|return|set|static|super|switch|this|throw|try|typeof|var|void|while|with|yield)\b/,
      'number': /\b(?:0[xX][\da-fA-F]+|0[bB][01]+|0[oO][0-7]+|NaN|Infinity)\b|(?:\b\d+(?:\.\d+)?|\B\.\d+)(?:[eE][+-]?\d+)?\b/,
      'function': /[_$a-zA-Z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\s*(?:\.\s*(?:apply|bind|call)\s*)?\()/,
      'operator': /--|\+\+|\*\*=?|=>|&&=?|\|\|=?|[!=]==|<<=?|>>>?=?|[-+*/%&|^!=<>]=?|\.{3}|\?\?=?|\?\.?|[~:]/
    });

    Prism.languages.javascript['class-name'][0].pattern = /(\b(?:class|interface|extends|implements|instanceof|new)\s+)[\w.\\]+/;

    Prism.languages.insertBefore('javascript', 'keyword', {
      'regex': {
        pattern: /(^|[^/])\/(?!\/)(\[[^\]\n\r\\]*(?:\\[\s\S][^\]\n\r\\]*)*\]|\\.|[^/\\\[\n\r])+\/[gimyus]{0,6}(?=\s*($|[\r\n,.;})\]]))/,
        lookbehind: true,
        greedy: true
      },
      'template-string': {
        pattern: /`(?:\\[\s\S]|\${(?:[^{}]|{(?:[^{}]|{[^}]*})*})+}|(?!\${)[^\\`])*`/,
        greedy: true,
        inside: {
          'template-punctuation': {
            pattern: /^`|`$/,
            alias: 'string'
          },
          'interpolation': {
            pattern: /\${(?:[^{}]|{(?:[^{}]|{[^}]*})*})+}/,
            inside: {
              'interpolation-punctuation': {
                pattern: /^\${|}$/,
                alias: 'punctuation'
              },
              rest: Prism.languages.javascript
            }
          },
          'string': /[\s\S]+/
        }
      }
    });

    Prism.languages.insertBefore('javascript', 'class-name', {
      'builtin': /\b(?:eval|arguments|Array|ArrayBuffer|Boolean|DataView|Date|Error|Function|Generator|GeneratorFunction|Infinity|JSON|Map|Math|NaN|Number|Object|Promise|Proxy|Reflect|RegExp|Set|String|Symbol|TypeError|WeakMap|WeakSet|console|document|window|globalThis|process|require|module|exports|__dirname|__filename|setTimeout|setInterval|clearTimeout|clearInterval)\b/
    });

    Prism.languages.js = Prism.languages.javascript;

    // ========== Language: Python ==========

    Prism.languages.python = {
      'comment': {
        pattern: /(^|[^\\])#.*/,
        lookbehind: true
      },
      'string': {
        pattern: /(?:[rR]|[uU]|[bB]|[fF]|[rR][fF]|[fF][rR])?(?:("""|''')[\s\S]*?\2|("|')(?:\\.|(?!\2)[^\\\r\n])*\2)/,
        greedy: true
      },
      'function': {
        pattern: /((?:^|\s)def\s+)[a-zA-Z_]\w*(?=\s*\()/,
        lookbehind: true
      },
      'class-name': {
        pattern: /(\bclass\s+)[a-zA-Z_]\w*(?=\s*[:\(])/,
        lookbehind: true
      },
      'keyword': /\b(?:False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield|self|cls)\b/,
      'builtin': /\b(?:abs|all|any|ascii|bin|bool|breakpoint|bytearray|bytes|callable|chr|classmethod|compile|complex|copyright|credits|delattr|dict|dir|divmod|enumerate|eval|exec|exit|filter|float|format|frozenset|getattr|globals|hasattr|hash|help|hex|id|input|int|isinstance|issubclass|iter|len|license|list|locals|map|max|memoryview|min|next|object|oct|open|ord|pow|print|property|quit|range|repr|reversed|round|set|setattr|slice|sorted|staticmethod|str|sum|super|tuple|type|vars|zip|__import__)\b/,
      'number': /\b(?:0[box][0-9a-fA-F]*|0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?[jJ]?\b/,
      'operator': /[-+%=]=?|!=|\*\*?=?|\/\/?=?|<[<=>]?|>[=>]?|[&|^~]/,
      'punctuation': /[{}[\];(),.:]/,
      'decorator': {
        pattern: /@[\w.]+/,
        alias: 'function'
      }
    };

    // ========== Language: C ==========

    Prism.languages.c = Prism.languages.extend('clike', {
      'keyword': /\b(?:_Alignas|_Alignof|_Atomic|_Bool|_Complex|_Generic|_Imaginary|_Noreturn|_Static_assert|_Thread_local|auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|goto|if|inline|int|long|register|restrict|return|short|signed|sizeof|static|struct|switch|typedef|union|unsigned|void|volatile|while|import|_Pragma)\b/,
      'builtin': /\b(?:printf|scanf|fprintf|sprintf|fopen|fclose|fread|fwrite|fseek|ftell|rewind|malloc|calloc|realloc|free|memcpy|memset|memmove|strcmp|strlen|strcpy|strcat|abs|sqrt|pow|sin|cos|tan|exp|log|ceil|floor|round|rand|srand|time|NULL|EOF)\b/,
      'number': /\b(?:0x[\da-f]+|0[0-7]*|\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:u|l|ul|lu|f|L)?\b/i
    });

    Prism.languages.insertBefore('c', 'function', {
      'macro': {
        pattern: /(^\s*)#\s*[a-z]+(?:[^\r\n\\/]|\/(?!\*)|\/\*(?:[^*]|\*(?!\/))*\*\/|\\(?:\r\n|[\s\S]))*/im,
        lookbehind: true,
        alias: 'property',
        inside: {
          'string': {
            pattern: /(#\s*include\s*)<[^>]+>/,
            lookbehind: true
          },
          'directive': {
            pattern: /(^#)\s*[a-z]+/,
            lookbehind: true,
            alias: 'keyword'
          },
          'macro-name': [
            {
              pattern: /(^#\s*define\s+)\w+/,
              lookbehind: true
            },
            {
              pattern: /(^#\s*(?:ifdef|ifndef|undef)\s+)\w+/,
              lookbehind: true
            }
          ]
        }
      }
    });

    // ========== Language: C++ ==========

    Prism.languages.cpp = Prism.languages.extend('c', {
      'keyword': /\b(?:alignas|alignof|asm|auto|bool|break|case|catch|char|char8_t|char16_t|char32_t|class|compl|concept|const|consteval|constexpr|const_cast|continue|co_await|co_return|co_yield|decltype|default|delete|do|double|dynamic_cast|else|enum|explicit|export|extern|float|for|friend|goto|if|inline|int|long|mutable|namespace|new|noexcept|nullptr|operator|private|protected|public|register|reinterpret_cast|requires|return|short|signed|sizeof|static|static_assert|static_cast|struct|switch|template|this|thread_local|throw|true|false|try|typedef|typeid|typename|union|unsigned|using|virtual|void|volatile|wchar_t|while|and|and_eq|bitand|bitor|not|not_eq|or|or_eq|xor|xor_eq)\b/,
      'builtin': /\b(?:std|vector|map|set|unordered_map|unordered_set|string|iostream|fstream|sstream|queue|stack|deque|list|array|tuple|pair|optional|variant|any|shared_ptr|unique_ptr|weak_ptr|make_shared|make_unique|move|forward|min|max|swap|sort|find|count|begin|end|cbegin|cend|size|empty|push_back|emplace_back|push_front|emplace_front|pop_back|pop_front|at|front|back|insert|erase|clear|reserve|resize|capacity|cout|cin|cerr|endl|flush|getline|stoi|stol|stoll|stof|stod|to_string|abs|sqrt|pow|sin|cos|tan|exp|log|ceil|floor|round|rand|srand|time|NULL|nullptr|EOF)\b/,
      'class-name': [
        {
          pattern: /(\b(?:class|struct|enum|union|typename)\s+)[~a-zA-Z_]\w*(?:\s*<[^>]*>)?/,
          lookbehind: true,
          inside: {
            'generic': {
              pattern: /<[^>]*>/,
              inside: {
                'punctuation': /[<>,:]/
              }
            },
            'punctuation': /~/
          }
        },
        {
          pattern: /(\b(?:new|delete)\s+)[~a-zA-Z_]\w*(?:\s*<[^>]*>)?(?=\s*[\(\[])/,
          lookbehind: true,
          inside: {
            'generic': {
              pattern: /<[^>]*>/,
              inside: {
                'punctuation': /[<>,:]/
              }
            },
            'punctuation': /~/
          }
        }
      ],
      'operator': /--|\+\+|->\*?|::|\.\*?|<<=?|>>=?|&&|\|\||<=|>=|==|!=|\+=|-=|\*=|\/=|%=|&=|\|=|\^=|[+\-*/%&|^~!=<>]=?|[?:]/
    });

    Prism.languages.insertBefore('cpp', 'string', {
      'raw-string': {
        pattern: /R"([^()\\ ]{0,16})\([\s\S]*?\)\1"/,
        greedy: true,
        alias: 'string'
      }
    });

    // ========== Language: Bash / Shell ==========

    Prism.languages.bash = {
      'shebang': {
        pattern: /^#!\s*\/.+/,
        alias: 'important'
      },
      'comment': {
        pattern: /(^|[^"{\\$])#.*/,
        lookbehind: true
      },
      'string': [
        {
          pattern: /(^|[^'"\\])"(?:\\[\s\S]|[^"\\\n])*"/,
          lookbehind: true,
          greedy: true
        },
        {
          pattern: /(^|[^'"\\])'(?:[^'\\\n])*'/,
          lookbehind: true,
          greedy: true
        }
      ],
      'variable': [
        /\$[\w#?$@!*\-]+/,
        /\${[^}]+}/
      ],
      'function': {
        pattern: /(^|[\s;|&])(?:[a-zA-Z_]\w*)(?=\s*\(\)\s*\{)/m,
        lookbehind: true
      },
      'keyword': /\b(?:if|then|else|elif|fi|for|in|do|done|while|until|case|esac|function|select|return|break|continue|exit|export|local|readonly|declare|typeset|let|eval|exec|source|trap|set|unset|shift|cd|echo|printf|read|test|[[|]]|\[\])\b/,
      'builtin': /\b(?:alias|bg|bind|builtin|caller|cd|command|compgen|complete|compopt|continue|dirs|disown|echo|enable|eval|exec|exit|false|fc|fg|getopts|hash|help|history|jobs|kill|let|local|logout|mapfile|popd|printf|pushd|pwd|read|readarray|readonly|return|set|shift|shopt|source|suspend|test|times|trap|true|type|typeset|ulimit|umask|unalias|unset|wait|cat|ls|rm|cp|mv|mkdir|rmdir|chmod|chown|grep|sed|awk|find|tar|gzip|gunzip|curl|wget|ssh|scp|git|make|sudo|su|man|ps|top|kill|nice|time|sleep|true|false|yes|head|tail|sort|uniq|cut|paste|join|split|wc|tr|tee|diff|patch|file|which|whereis|whoami|date|cal|bc|expr|seq|du|df|free|uname|uptime|hostname|ifconfig|ping|netstat|ss)\b/,
      'number': /\b\d+(?:\.\d+)?\b/,
      'operator': /[|&;<>()$`\\'"]|&&|\|\||<<|>>|;;|[<>]&/,
      'punctuation': /[{}[\];(),.:]/,
      'property': /^[a-zA-Z_][\w-]*(?==)/m
    };

    Prism.languages.shell = Prism.languages.bash;

    // ========== Language: JSON ==========

    Prism.languages.json = {
      'property': {
        pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"(?=\s*:)/,
        lookbehind: true,
        greedy: true
      },
      'string': {
        pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"(?!\s*:)/,
        lookbehind: true,
        greedy: true
      },
      'comment': {
        pattern: /\/\/.*|\/\*[\s\S]*?(?:\*\/|$)/,
        greedy: true
      },
      'number': /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/,
      'punctuation': /[{}[\],:]/,
      'boolean': /\b(?:true|false)\b/,
      'null': /\bnull\b/
    };

    // ========== Language: Markdown ==========

    Prism.languages.markdown = {
      'title': [
        {
          pattern: /^#{1,6}.+/m,
          inside: {
            'punctuation': /^#+/
          }
        },
        {
          pattern: /\S.+\n=+\s*$|\S.+\n-+\s*$/,
          inside: {
            'string': /^.+(?=\n[=-]+$)/m,
            'punctuation': /[=-]+$/
          }
        }
      ],
      'hr': /^-{3,}\s*$/m,
      'blockquote': /^>.+/m,
      'code': [
        /```[\s\S]*?```/,
        /``.+?``|`[^`]+`/
      ],
      'bold': {
        pattern: /(^|[^*])\*\*(?!\s)(?:[^*\n]|\*(?!\*))+?\*\*/,
        lookbehind: true
      },
      'italic': {
        pattern: /(^|[^*])\*(?!\s)(?:[^*\n])+?\*(?!\*)/,
        lookbehind: true
      },
      'url': /\[[^\]]*\]\([^)]*\)/,
      'list': {
        pattern: /^(\s*)[-*+]\s+/m,
        lookbehind: true
      },
      'table': /\|.*\|/,
      'strike': /~~.+?~~/
    };

    Prism.languages.md = Prism.languages.markdown;

    // ========== Language: SQL ==========

    Prism.languages.sql = {
      'comment': [
        /--.*/,
        /\/\*[\s\S]*?\*\//
      ],
      'string': {
        pattern: /'(?:[^']|'')*'/,
        greedy: true
      },
      'variable': [
        /@[\w.]+/,
        /@@[\w.]+/
      ],
      'function': /\b(?:AVG|COUNT|SUM|MIN|MAX|FIRST|LAST|UCASE|LCASE|MID|LEN|ROUND|NOW|FORMAT|COALESCE|IFNULL|ISNULL|NVL|CONCAT|SUBSTRING|TRIM|UPPER|LOWER|CAST|CONVERT|DATE_ADD|DATE_SUB|DATEDIFF|EXTRACT|DATE_FORMAT|STR_TO_DATE|ASCII|CHAR|CHAR_LENGTH|CHARINDEX|DIFFERENCE|PATINDEX|REPLACE|REPLICATE|REVERSE|SOUNDEX|SPACE|STUFF|UNICODE|USER_NAME|DATABASE)\b/i,
      'keyword': /\b(?:ADD|ALL|ALTER|AND|ANY|AS|ASC|BACKUP|BETWEEN|BY|CASE|CHECK|COLUMN|CONSTRAINT|CREATE|DATABASE|DEFAULT|DELETE|DESC|DISTINCT|DROP|ELSE|END|EXCEPT|EXISTS|FOREIGN|FROM|FULL|GROUP|HAVING|IN|INDEX|INNER|INSERT|INTO|IS|JOIN|LEFT|LIKE|LIMIT|NOT|NULL|ON|OR|ORDER|OUTER|PRIMARY|PROCEDURE|REPLACE|RETURNS|REVOKE|RIGHT|ROWNUM|SELECT|SET|TABLE|THEN|TO|TOP|TRUNCATE|UNION|UNIQUE|UPDATE|VALUES|VIEW|WHEN|WHERE|WITH|WITHOUT|BEGIN|DECLARE|EXEC|EXECUTE|FETCH|INTO|OPEN|CLOSE|DEALLOCATE|CURSOR|WHILE|BREAK|CONTINUE|GOTO|RETURN|TRY|CATCH|THROW|RAISERROR|COMMIT|ROLLBACK|SAVEPOINT|TRANSACTION|GRANT|REVOKE|DENY|IF|ELSE|CASCADE|RESTRICT|NO ACTION|SET NULL|SET DEFAULT)\b/i,
      'boolean': /\b(?:TRUE|FALSE|UNKNOWN)\b/i,
      'number': /\b-?(?:0x[\da-f]+|\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b/i,
      'operator': /[-+*\/%=<>!&|~^]|<>|<=|>=|!=|&&|\|\||<<|>>/,
      'punctuation': /[{}[\];(),.:]/,
      'data-type': /\b(?:INT|INTEGER|BIGINT|SMALLINT|TINYINT|BIT|DECIMAL|NUMERIC|FLOAT|REAL|MONEY|SMALLMONEY|DATE|TIME|DATETIME|DATETIME2|SMALLDATETIME|DATETIMEOFFSET|TIMESTAMP|CHAR|VARCHAR|TEXT|NCHAR|NVARCHAR|NTEXT|BINARY|VARBINARY|IMAGE|CLOB|BLOB|JSON|XML|UUID|SERIAL|BIGSERIAL|BOOLEAN|BOOL|ARRAY|ENUM|SET)\b/i
    };

    // Auto-highlight on DOMContentLoaded
    if (typeof document !== 'undefined') {
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
          Prism.highlightAll();
        });
      } else {
        Prism.highlightAll();
      }
    }

    return Prism;

  })();

  global.Prism = Prism;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
