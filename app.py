from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import bz2
import os
import numpy as np

# path="C:\\Users\\HP\\Desktop\\repos\\articles_category_predition"
# os.chdir(path)

# load the model from disk
fp = bz2.open('tfidf_2020_small_bz2.pkl','rb')
transform_tfidf_model = pickle.load(fp)
fp.close()

xgboost_predict = pickle.load(open('xgboost_2020_small.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Body = request.form['message']
        Headline = request.form['headline']
        headline = [Headline]
        body = [Body]
        all_articles = pd.DataFrame({'_source.headline': headline, '_source.body': body})
        all_articles['headline_body'] = all_articles['_source.headline'] + " " + all_articles['_source.body']
        all_articles['headline_body'] = all_articles['headline_body'].str.lower()
        all_articles['headline_body'] = all_articles['headline_body'].str.replace('http\S+|www.\S+', '', case=False)
        all_articles['headline_body'] = all_articles['headline_body'].str.replace("[^a-z-']+", " ")
        all_articles['headline_body'] = all_articles['headline_body'].str.replace("\s+", " ")

        all_articles['headline_body'] = all_articles['headline_body'].str.split()

        # all_articles['word_length'] = all_articles['headline_body'].str.len()
        # all_articles = all_articles[all_articles['word_length'] > 20]

        def remove_short_words(text):
            words = [w for w in text if len(w) > 2]
            return words

        all_articles['headline_body'] = all_articles['headline_body'].apply(lambda x: remove_short_words(x))

        new_words = ["div", "style", "nbsp", "font", "http", "bodytext", "class", "href", "rdquo", "ldquo", "an",
                     "rsquo", "news", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab",
                     "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act",
                     "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after",
                     "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows",
                     "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
                     "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore",
                     "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear",
                     "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise",
                     "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av",
                     "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc",
                     "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand",
                     "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside",
                     "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn",
                     "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1",
                     "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd",
                     "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly",
                     "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider",
                     "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt",
                     "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx",
                     "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described",
                     "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do",
                     "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds",
                     "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee",
                     "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere",
                     "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es",
                     "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody",
                     "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2",
                     "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first",
                     "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former",
                     "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full",
                     "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give",
                     "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr",
                     "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has",
                     "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll",
                     "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's",
                     "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither",
                     "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu",
                     "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd",
                     "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate",
                     "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate",
                     "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into",
                     "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll",
                     "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just",
                     "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2",
                     "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les",
                     "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj",
                     "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2",
                     "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime",
                     "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss",
                     "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu",
                     "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely",
                     "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't",
                     "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj",
                     "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not",
                     "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain",
                     "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay",
                     "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or",
                     "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves",
                     "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3",
                     "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc",
                     "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm",
                     "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly",
                     "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides",
                     "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra",
                     "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref",
                     "refs", "regarding", "regardless", "regards", "related", "relatively", "research",
                     "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right",
                     "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa",
                     "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly",
                     "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves",
                     "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't",
                     "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've",
                     "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly",
                     "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn",
                     "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes",
                     "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify",
                     "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially",
                     "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1",
                     "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th",
                     "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the",
                     "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
                     "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto",
                     "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're",
                     "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou",
                     "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti",
                     "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards",
                     "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve",
                     "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under",
                     "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur",
                     "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v",
                     "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols",
                     "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't",
                     "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren",
                     "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence",
                     "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's",
                     "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever",
                     "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely",
                     "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't",
                     "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf",
                     "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl",
                     "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves",
                     "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]

        def remove_stopwords(text):
            words = [w for w in text if w not in new_words]
            return words

        all_articles['headline_body'] = all_articles['headline_body'].apply(lambda x: remove_stopwords(x))

        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()

        def word_lemmatizer(text):
            lem_text = " ".join([lemmatizer.lemmatize(w) for w in text])
            return lem_text

        all_articles['headline_body'] = all_articles['headline_body'].apply(lambda x: word_lemmatizer(x))

        transform = transform_tfidf_model.transform(all_articles['headline_body'])
        predict_proba = pd.DataFrame(xgboost_predict.predict_proba(transform))

        predict_proba.columns = ['auto', 'tech', 'politics', 'business', 'sports', 'lifestyle', 'entertainment',
                                 'logistics', 'economy', 'markets', 'education', 'health', 'startup', 'travel',
                                 'gaming', 'crime', 'banking', 'food', 'finance']

        def get_n_largest_ind(a, n):
            ind = np.argpartition(a, -n)[-n:]
            return ind[0]

        cols = predict_proba.columns
        predict_proba['top1'] = predict_proba[cols].apply(lambda x: cols[get_n_largest_ind(x, 1)], axis=1)
        predict_proba['top2'] = predict_proba[cols].apply(lambda x: cols[get_n_largest_ind(x, 2)], axis=1)

        rlt = predict_proba[['top1', 'top2']]

        predict_proba = predict_proba.drop(['top1', 'top2'], axis=1)

        test_1 = pd.DataFrame(np.sort(predict_proba.values)[:, -2:], columns=['2nd-largest', 'largest'])

        test_1['category_1'] = np.where(test_1['largest'] >= 0.5, test_1['largest'], "others")
        test_1['category_2'] = np.where(
            (test_1['largest'] < 0.5) & (test_1['largest'] >= 0.25) & (test_1['2nd-largest'] < 0.5) & (
                    test_1['2nd-largest'] >= 0.25), test_1['largest'], "others")
        test_1['category_3'] = np.where(
            (test_1['largest'] < 0.5) & (test_1['largest'] >= 0.25) & (test_1['2nd-largest'] < 0.5) & (
                    test_1['2nd-largest'] >= 0.25), test_1['2nd-largest'], "others")
        test_1['category_4'] = np.where((test_1['largest'] >= 0.5) & (test_1['2nd-largest'] >= 0.25),
                                        test_1['2nd-largest'], "others")

        top_prob = pd.merge(test_1, rlt, left_index=True, right_index=True)

        top_prob['final_cat_1'] = np.where(top_prob['category_1'] != "others", top_prob['top1'],
                                           np.where(top_prob['category_2'] != "others", top_prob['top1'], "others"))
        top_prob['final_cat_2'] = np.where(top_prob['category_3'] != "others", top_prob['top2'],
                                           np.where(top_prob['category_4'] != "others", top_prob['top2'], ""))

        final_article = pd.merge(all_articles, top_prob[['final_cat_1', 'final_cat_2']], left_index=True,
                                 right_index=True)
        final_article['ML_category'] = final_article['final_cat_1'] + ', ' + final_article['final_cat_2']

        # final_article['Advertising_and_Marketing'] = np.where((all_articles['_source.body'].str.contains(r'\bad\b|\bmarketing\b', case = False)) & (all_articles['ML_category'].str.contains(r'\bothers\b')), 'Advertising_and_Marketing', "")

        final_article['Agriculture'] = np.where((final_article['_source.headline'].str.contains(
            r'\bagriculture\b|\bfarming\b|\bfarmers\b|\bagritech\b|\bagri-tech\b', case=False, na=False)) & (
                                                    final_article['ML_category'].str.contains(
                                                        r'\bbusiness\b|\bothers\b', na=False)), 'Agriculture', "")

        # final_article['Arts_and_Culture'] = np.where((all_articles['_source.body'].str.contains(r'\bagriculture\b|\bfarming\b|\bfarmers\b|\bagritech\b', case = False)) & (all_articles['ML_category'].str.contains(r'\bbusiness\b|\bothers\b'), 'Agriculture', "")

        final_article['Aviation'] = np.where((final_article['_source.headline'].str.contains(
            r'\bflights\b|\bairline\b|\bindigo\b|\bjet airways\b|\bvistara\b|\bair asia\b|\bgo air\b|\bair india\b|\baviation\b',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bothers\b|\bmarkets\b|\bfinance\b', na=False)), 'Aviation', "")

        final_article['Blockchain'] = np.where((final_article['_source.body'].str.contains(
            r'\bbitcoin\b|\bcrypto\b|\bethereum\b|\blitecoin\b|\bblockchain\b', case=False, na=False)) & (
                                                   final_article['ML_category'].str.contains(
                                                       r'\bbusiness\b|\bothers\b|\bmarkets\b|\bfinance\b|\bbanking\b',
                                                       na=False)), 'Blockchain', "")

        final_article['Consumer_Tech'] = np.where((final_article['_source.headline'].str.contains(
            r'gadget|mobile|laptop|electronics|wearables|consumer-tech|consumer tech', case=False, na=False)) & (
                                                      final_article['ML_category'].str.contains(
                                                          r'\bbusiness\b|\btech\b|\bstartup\b', na=False)),
                                                  'Consumer_Tech', "")

        final_article['Covid_Electronic'] = np.where(
            final_article['_source.body'].str.contains(r'corona virus|covid-19|\bcovid\b|covid - 19|coronavirus',
                                                       case=False, na=False), 'Covid_Electronic', "")

        final_article['Covid_Online'] = np.where(
            final_article['_source.body'].str.contains(r'corona virus|covid-19|\bcovid\b|covid - 19|coronavirus',
                                                       case=False, na=False), 'Covid_Online', "")

        final_article['Cryptocurrency'] = np.where((final_article['_source.body'].str.contains(
            r'\bbitcoin\b|\bcrypto\b|\bethereum\b|\blitecoin\b', case=False, na=False)) & (
                                                       final_article['ML_category'].str.contains(
                                                           r'\bbusiness\b|\bothers\b|\bmarkets\b|\bfinance\b|\bbanking\b',
                                                           na=False)), 'Cryptocurrency', "")

        final_article['E_commerce'] = np.where((final_article['_source.headline'].str.contains(
            r'flipkart|e-commerce|\bmyntra\b|\bjabong\b|snapdeal|\balibaba\b', case=False, na=False)) & (
                                                   final_article['ML_category'].str.contains(
                                                       r'\bbusiness\b|\blogistics\b|\bstartup\b|\btech\b|\bothers\b',
                                                       na=False)), 'E_Commerce', "")

        final_article['Fintech'] = np.where((final_article['_source.body'].str.contains(
            r'fintech|payment gateway|banking technologies|fin-tech', case=False, na=False)) & (
                                                final_article['ML_category'].str.contains(
                                                    r'\bbusiness\b|\bfinance\b|\bstartup\b|\btech\b|\bbanking\b',
                                                    na=False)), 'Fintech', "")

        final_article['FMCG'] = np.where((final_article['_source.headline'].str.contains(
            r'\bfmcg\b|fast moving consumer goods|\bhindustan unilever\b|\bcolgate-palmolive\b|\bitc\b|\bnestle\b|\bparle\b|\bbritannia\b|\bmarico\b|\bp&g\b|\bprocter and gamble\b|\bgodrej\b|\bamul\b|\bhul\b',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bfood\b|\blogistics\b|\bothers\b', na=False)), 'FMCG', "")

        final_article['HR'] = np.where((final_article['_source.headline'].str.contains(
            r'\brecruitment\b|human resource|\bhiring\b|\bhrm\b|\bhr\b|\bhire\b|\bhired\b', case=False, na=False)) & (
                                           final_article['ML_category'].str.contains(r'\beducation\b|\bothers\b',
                                                                                     na=False)), 'HR', "")

        final_article['Infrastructure'] = np.where((final_article['_source.headline'].str.contains(
            r'\bconstruction\b|\bconstructing\b', case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bauto\b|\bbanking\b|\blogistics\b|\bmarkets\b|\bothers\b', na=False)), 'Infrastructure', "")

        final_article['IT'] = np.where((final_article['_source.headline'].str.contains(
            r'\btcs\b|\btata consultancy services\b|\binfosys\b|\bwipro\b|\bhcl\b|\btech mahindra\b|\boracle\b|\bmindtree\b|\bit industry\b|\bit industries\b',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\btech\b|\bfinance\b|\bmarkets\b|\beconomy\b|\bothers\b', na=False)), 'IT', "")

        final_article['Legal'] = np.where((final_article['_source.headline'].str.contains(
            r'\baffidavit\b|\bcompliance\b|\blegal notice\b', case=False, na=False)) & (
                                              final_article['ML_category'].str.contains(
                                                  r'\bcrime\b|\bbusiness\b|\beconomy\b|\bothers\b', na=False)), 'Legal',
                                          "")

        final_article['Retail'] = np.where((final_article['_source.headline'].str.contains(
            r'\bwalmart\b|\bretail\b|\bdmart\b|\bfuture group\b|\bhul\b|\bp&g\b|\bpatanjali\b|\bitc\b|\bbig bazaar\b|\breliance trends\b|\bmaxx\b|\bshoppers stop\b|\bpantaloons\b',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bfood\b|\blogistics\b|\bothers\b', na=False)), 'Retail', "")

        final_article['SaaS'] = np.where((final_article['_source.headline'].str.contains(
            r'\bsaas\b|\bsoftware as a service\b|\bbusiness app\b', case=False, na=False)) & (
                                             final_article['ML_category'].str.contains(
                                                 r'\btech\b|\bbusiness\b|\bfinance\b|\bstartup\b|\bothers\b',
                                                 na=False)), 'SaaS', "")

        final_article['Telecom'] = np.where((final_article['_source.headline'].str.contains(
            r'\bairtel\b|\bverizon\b|\bvodafone\b|\bjio\b|\bbsnl\b', case=False, na=False)) & (
                                                final_article['ML_category'].str.contains(r'\bbusiness\b|\bothers\b',
                                                                                          na=False)), 'Telecom', "")

        final_article['VC'] = np.where((final_article['_source.body'].str.contains(
            r'\bventure capitalists\b|\bvc\b|\bventure capitalist\b', case=False, na=False)) & (
                                           final_article['ML_category'].str.contains(
                                               r'\btech\b|\bbusiness\b|\bfinance\b|\bstartup\b|\bothers\b', na=False)),
                                       'VC', "")

        final_article['final_cat_1'] = final_article['final_cat_1'].str.replace('others', "")

        unmerged_columns = ['index', '_id', '_index', '_type', '_score', '_source.domain',
                            '_source.url', '_source.updated_at', '_source.type', '_source.headline',
                            '_source.published_at', '_source.body', '_source.section',
                            '_source.top_image_url', '_source.authors', '_source.isTestSample',
                            '_source.lang', 'headline_body', 'ML_category', 'journalist_id', 'name']

        merged_columns = []
        for i in final_article.columns:
            if i not in unmerged_columns:
                merged_columns.append(i)

        final_article['list'] = final_article[merged_columns].values.tolist()
        final_article['all_category'] = final_article['list'].apply(' '.join)
        final_article['all_category'] = final_article['all_category'].str.strip()
        final_article['all_category'] = final_article['all_category'].str.replace('\s+', ', ')

        final_article['word_count'] = final_article['all_category'].str.len()

        final_article['all_category'] = np.where(final_article['word_count'] == 0, 'city_and_general',
                                                 final_article['all_category'])

        all_categories = final_article['all_category'][0]
        # category_1 = top_prob['final_cat_1'][0]
        # category_2 = top_prob['final_cat_2'][0]
        top1 = top_prob['top1'][0]
        top2 = top_prob['top2'][0]
        second_largest = round(top_prob['2nd-largest'][0], ndigits=2)
        largest = round(top_prob['largest'][0], ndigits=2)
        category = 'Categories are: {}'.format(all_categories) + '. ' + 'For {} the probabity is {}'.format(top1,
                                                                                                            largest) + ' and ' + 'for {} the probabity is {}'.format(
            top2, second_largest)

    return render_template('home.html', prediction=category)


if __name__ == '__main__':
    app.run(debug=True)