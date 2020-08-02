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
        all_articles['headline_body'] = all_articles['_source.headline'] + " " + \
                                                 all_articles['_source.body']
        all_articles['headline_body'] = all_articles['headline_body'].str.lower()
        all_articles['headline_body'] = all_articles['headline_body'].str.replace(
            r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', ' ')
        all_articles['headline_body'] = all_articles['headline_body'].str.replace('\r|\n', ' ')
        all_articles['headline_body'] = all_articles['headline_body'].str.replace('http\S+|www.\S+',
                                                                                                    ' ', case=False)
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

        final_article['agriculture_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'\bagriculture\b|\bfarming\b|farmer|\bagritech\b|\bagri-tech\b|\bcotton\b|\bcrop\b|\brajma\b|quintal|cotton seed|cotton yarn|cotton price|seed price|\bgreen pea\b|rajma price|seed production|basmati rice|cotton association|\bagri\b',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bothers\b|\bmarkets\b|\bfinance\b|\beconomy\b', na=False)), 'agriculture', "")
        final_article['agriculture'] = np.where((final_article['_source.body'].str.contains(
            r'cotton seed|cotton yarn|cotton price|seed price|rajma price|indian cotton|seed production|basmati rice|cotton association|cargill india',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bothers\b|\bmarkets\b|\bfinance\b|\beconomy\b', na=False)), 'agriculture',
                                                final_article['agriculture_headline'])

        final_article['oil_and_energy_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'\bongc\b|aramco|bpcl|reliance petroleum|\bgail\b|petrol|diesel|\bioc\b|\bbhel\b|\bntpc\b|\boil\b|\benergy\b|indian oil|\btotal s.a\b|royal dutch shell',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bmarkets\b|\bfinance\b|\beconomy\b', na=False)), 'oil_and_energy', "")
        final_article['oil_and_energy'] = np.where((final_article['_source.body'].str.contains(
            r'KG-D6|krishna godavari basin|petroleum ministry|\blpg\b|gasoline|bharat gas|indiane', case=False,
            na=False)) & (final_article['ML_category'].str.contains(r'\bbusiness\b|\bmarkets\b|\bfinance\b|\beconomy\b',
                                                                    na=False)), 'oil_and_energy',
                                                   final_article['oil_and_energy_headline'])

        final_article['aviation_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'air Charters|air india|aviation|indian airlines|paramount airways|go air|kingfisher airlines|spice jet|air sahara|jet airways|vistara|flight|virgin airlines|spaceX',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bothers\b|\bmarkets\b|\bfinance\b|\beconomy\b', na=False)), 'aviation', "")
        final_article['aviation'] = np.where(
            (final_article['_source.body'].str.contains(r'aviation', case=False, na=False)) & (
                final_article['ML_category'].str.contains(r'\bbusiness\b|\bothers\b|\bmarkets\b|\bfinance\b|\beconomy',
                                                          na=False)), 'aviation', final_article['aviation_headline'])

        final_article['blockchain'] = np.where((final_article['_source.body'].str.contains(
            r'\bbitcoin\b|\bcrypto\b|\bethereum\b|\blitecoin\b|\bblockchain\b|\bxrp\b|\bZebpay\b', case=False,
            na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bothers\b|\bmarkets\b|\bfinance\b|\bbanking\b|\btech\b|\bstartup\b', na=False)),
                                               'blockchain', "")

        final_article['cybersecurity'] = np.where((final_article['_source.body'].str.contains(
            r'hacker|encryption|decryption|encrypted|decrypted|cybersecurity|hacked|cybercrime|cyber crime', case=False,
            na=False)) & (final_article['ML_category'].str.contains(r'\bbusiness\b|\btech\b', na=False)),
                                                  'cybersecurity', "")

        final_article['consumertech_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'gadget|mobile|laptop|electronic|wearables|samsung galaxy|iphone|google pixel|xiaomi|realme|speaker|\btv\b|smart watch|smartphone|earphones|bezeless',
            case=False, na=False)) & (final_article['ML_category'].str.contains(r'\bbusiness\b|\btech\b|\bstartup\b',
                                                                                na=False)), 'consumer_tech', "")
        final_article['consumer_tech'] = np.where((final_article['_source.body'].str.contains(
            r'snapdragon|mediatek|exynos|intel processor|inch display|amd radeon|smart watch|wireless earphones|smart tv',
            case=False, na=False)) & (final_article['ML_category'].str.contains(r'\bbusiness\b|\btech\b|\bstartup\b',
                                                                                na=False)), 'consumer_tech',
                                                  final_article['consumertech_headline'])

        # final_article['Covid_Electronic'] = np.where(final_article['_source.body'].str.contains(r'corona virus|covid-19|\bcovid\b|covid - 19|coronavirus', case = False, na = False), 'Covid_Electronic', "")

        # final_article['Covid_Online'] = np.where(final_article['_source.body'].str.contains(r'corona virus|covid-19|\bcovid\b|covid - 19|coronavirus', case = False, na = False), 'Covid_Online', "")

        # final_article['Cryptocurrency'] = np.where((final_article['_source.body'].str.contains(r'\bbitcoin\b|\bcrypto\b|\bethereum\b|\blitecoin\b', case = False, na = False)) & (final_article['ML_category'].str.contains(r'\bbusiness\b|\bothers\b|\bmarkets\b|\bfinance\b|\bbanking\b', na = False)), 'Cryptocurrency', "")

        final_article['ecommerce_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'flipkart|e-commerce|ecommerce|myntra|jabong|snapdeal|alibaba|commerce industry|ecommerce company|ali baba|ebay|india mart|indiamart|justdial|makemytrip|bookmyshow|1mg',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\blogistics\b|\bstartup\b|\btech\b|\bothers\b', na=False)), 'e_commerce', "")
        final_article['e_commerce'] = np.where((final_article['_source.body'].str.contains(
            r'commerce industry|ecommerce company|ecommerce', case=False, na=False)) & (
                                                   final_article['ML_category'].str.contains(
                                                       r'\bbusiness\b|\blogistics\b|\bstartup\b|\btech\b|\bothers\b',
                                                       na=False)), 'e_commerce', final_article['ecommerce_headline'])

        final_article['fintech_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'zerodha|paytm|googlepay|phonepe|phone-pe|\bupi\b|google-pay', case=False, na=False)) & (
                                                         final_article['ML_category'].str.contains(
                                                             r'\bbusiness\b|\bfinance\b|\bstartup\b|\btech\b|\bbanking\b',
                                                             na=False)), 'fintech', "")
        final_article['fintech'] = np.where((final_article['_source.body'].str.contains(
            r'payment gateway|digital payment|payment service|upi payment|fintech|fin-tech|Digital Transaction|Digital Wallet',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bfinance\b|\bstartup\b|\btech\b|\bbanking\b', na=False)), 'fintech',
                                            final_article['fintech_headline'])

        final_article['fmcg_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'\bfmcg\b|fast moving consumer goods|\bhindustan unilever\b|\bcolgate-palmolive\b|\bitc\b|\bnestle\b|\bparle\b|\bbritannia\b|\bmarico\b|\bp&g\b|\bprocter and gamble\b|\bamul\b|\bhul\b|\bpatanjali\b|cargill india|\bdabur\b|\bgodrej\b',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bfood\b|\blogistics\b|\bothers\b|\bfinance\b|\bmarkets\b|\bstartup\b', na=False)), 'fmcg',
                                                  "")
        final_article['fmcg'] = np.where((final_article['_source.body'].str.contains(
            r'\bfmcg\b|fast moving consumer goods|Food processing|kirana|FSSAI|mom and pop|hyperlocal', case=False,
            na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bfood\b|\blogistics\b|\bothers\b|\bfinance\b|\bmarkets\b|\bstartup\b', na=False)), 'fmcg',
                                         final_article['fmcg_headline'])

        final_article['hr_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'recruitment|human resource|\bhiring\b|\bhr\b|layoff|final placement|company hire|salary hike|average salary|hire people|recruitment company|increasing hiring|increase hiring',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\beducation\b|\bothers\b|\btech\b|\bstartup\b|\bauto\b|\bbanking\b|\bbusiness\b|\beconomy\b|\bfinance\b|\blogistics\b|\bmarkets\b',
            na=False)), 'hr', "")
        final_article['hr'] = np.where((final_article['_source.body'].str.contains(
            r'final placement|company hire|salary hike|average salary|hire people|recruitment company|increasing hiring|increase hiring',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\beducation\b|\bothers\b|\btech\b|\bstartup\b|\bauto\b|\bbanking\b|\bbusiness\b|\beconomy\b|\bfinance\b|\blogistics\b|\bmarkets\b',
            na=False)), 'hr', final_article['hr_headline'])

        final_article['infrastructure'] = np.where((final_article['_source.body'].str.contains(
            r'\bconstruction\b|\bconstructing\b|Toll Tax|Circle rate|special economic zone|\bsez\b|\bbhk\b|Ministry of Road and Transportation|Border Roads Organization|\bNHAI\b|Real Estate|Lodha Group|Essel Infra',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bauto\b|\bbanking\b|\blogistics\b|\bmarkets\b|\bothers\b|\bbusiness\b|\beconomy\b|\bfinance\b',
            na=False)), 'infrastructure', "")

        final_article['it_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'\btcs\b|\btata consultancy services\b|\binfosys\b|\bwipro\b|\bhcl\b|\btech mahindra\b|\boracle\b|\bmindtree\b|\bit industry\b|\bit industries\b',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\btech\b|\bfinance\b|\bmarkets\b|\beconomy\b|\bothers\b', na=False)), 'IT', "")
        final_article['IT'] = np.where(
            (final_article['_source.body'].str.contains(r'it industry|it industries', case=False, na=False)) & (
                final_article['ML_category'].str.contains(r'\btech\b|\bfinance\b|\bmarkets\b|\beconomy\b|\bothers\b',
                                                          na=False)), 'IT', final_article['it_headline'])

        # final_article['Legal'] = np.where((final_article['_source.body'].str.contains(r'\baffidavit\b|\bcompliance\b|\blegal notice\b', case = False, na = False)) & (final_article['ML_category'].str.contains(r'\bcrime\b|\bbusiness\b|\beconomy\b|\bothers\b', na = False)), 'legal', "")

        # final_article['Retail'] = np.where((final_article['_source.headline'].str.contains(r'\bwalmart\b|\bretail\b|\bdmart\b|\bfuture group\b|\bhul\b|\bp&g\b|\bpatanjali\b|\bitc\b|\bbig bazaar\b|\breliance trends\b|\bmaxx\b|\bshoppers stop\b|\bpantaloons\b', case = False, na = False)) & (final_article['ML_category'].str.contains(r'\bbusiness\b|\bfood\b|\blogistics\b|\bothers\b', na = False)), 'Retail', "")

        final_article['SaaS_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'\bSaaS\b|Freshworks|\bZoho\b|Haptik|Hubspot|Slack|Shopify|MailChimp|Chargebee', case=False, na=False)) & (
                                                      final_article['ML_category'].str.contains(
                                                          r'\btech\b|\bbusiness\b|\bfinance\b|\bstartup\b|\bothers\b',
                                                          na=False)), 'SaaS', "")
        final_article['SaaS'] = np.where((final_article['_source.body'].str.contains(
            r'\bSaaS\b|chatbot|cloud based software|cloud computing', case=False, na=False)) & (
                                             final_article['ML_category'].str.contains(
                                                 r'\btech\b|\bbusiness\b|\bfinance\b|\bstartup\b|\bothers\b',
                                                 na=False)), 'SaaS', final_article['SaaS_headline'])

        final_article['telecom_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'prepaid plan|vodafone idea|5g network|\bjio\b|recharge plan|airtel|vodafone|\btelecom\b|unlimited data',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bothers\b|\beconomy\b|\bfinance\b', na=False)), 'telecom', "")
        final_article['telecom'] = np.where(
            (final_article['_source.body'].str.contains(r'prepaid plan|recharge plan', case=False, na=False)) & (
                final_article['ML_category'].str.contains(r'\bbusiness\b|\bothers\b|\beconomy\b|\bfinance\b',
                                                          na=False)), 'telecom', final_article['telecom_headline'])

        final_article['VC_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'venture capitalists|\bvc\b|funding alert|venture capitalists|funding round|sequoia capital|vision fund|angel investor|raised funding|startup raised|capital investor',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\btech\b|\bbusiness\b|\bfinance\b|\bstartup\b|\bothers\b', na=False)), 'VC', "")
        final_article['VC'] = np.where((final_article['_source.body'].str.contains(
            r'funding alert|venture capitalists|funding round|venture capital|sequoia capital|vision fund|angel investor|raised funding|startup raised|capital investor',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\btech\b|\bbusiness\b|\bfinance\b|\bstartup\b|\bothers\b', na=False)), 'VC', final_article['VC_headline'])

        final_article['logistics_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'logistics|agarwal packers|blue dart|container corporation|dhl express|fedex express|first flight|gati ltd|transport corporation',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bfinance\b|\bstartup\b|\bothers\b|\bmarkets\b|\beconomy\b', na=False)) & (
                                                           ~final_article['ML_category'].str.contains(r'\blogistics\b',
                                                                                                      na=False)),
                                                       'logistics', "")
        final_article['logistics'] = np.where(
            (final_article['_source.body'].str.contains(r'logistics', case=False, na=False)) & (
                final_article['ML_category'].str.contains(
                    r'\bbusiness\b|\bfinance\b|\bstartup\b|\bothers\b|\bmarkets\b|\beconomy\b', na=False)) & (
                ~final_article['ML_category'].str.contains(r'\blogistics\b', na=False)), 'logistics',
            final_article['logistics_headline'])

        final_article['startup_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'startup|start-up|funding alert|funding round|private equity|series round|venture capital firm|vision fund|crowdfunding',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bfinance\b|\bothers\b|\bmarkets\b', na=False)) & (
                                                         ~final_article['ML_category'].str.contains(r'\bstartup\b',
                                                                                                    na=False)),
                                                     'startup', "")
        final_article['startup'] = np.where((final_article['_source.body'].str.contains(
            r'funding alert|funding round|private equity|series round|venture capital firm|vision fund|startup',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bfinance\b|\bothers\b|\bmarkets\b', na=False)) & (
                                                ~final_article['ML_category'].str.contains(r'\bstartup\b', na=False)),
                                            'startup', final_article['startup_headline'])

        final_article['travel_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'cruise|travel to|travel advice|places to visit|best trip', case=False, na=False)) & (
                                                        final_article['ML_category'].str.contains(
                                                            r'\bbusiness\b|\bfinance\b|\bothers\b|\bentertainment\b|\blifestyle\b',
                                                            na=False)) & (
                                                        ~final_article['ML_category'].str.contains(r'\btravel\b',
                                                                                                   na=False)), 'travel',
                                                    "")
        final_article['travel'] = np.where(
            (final_article['_source.body'].str.contains(r'places to visit|best trip', case=False, na=False)) & (
                final_article['ML_category'].str.contains(
                    r'\bbusiness\b|\bfinance\b|\bothers\b|\bentertainment\b|\blifestyle\b', na=False)) & (
                ~final_article['ML_category'].str.contains(r'\btravel\b', na=False)), 'travel',
            final_article['travel_headline'])

        final_article['food_headline'] = np.where((final_article['_source.headline'].str.contains(
            r'recipe|healthy eating|healthy diet|\bdiet\b|fruit|vegetable|\bmeal\b|beverage|\bfood\b|breakfast|lunch|restaurant|brunch|snacks|eating',
            case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bothers\b|\bentertainment\b|\blifestyle\b|\bhealth\b', na=False)) & (
                                                      ~final_article['ML_category'].str.contains(r'\bfood\b',
                                                                                                 na=False)), 'food', "")
        final_article['food'] = np.where((final_article['_source.body'].str.contains(
            r'recipe|healthy eating|healthy diet', case=False, na=False)) & (final_article['ML_category'].str.contains(
            r'\bbusiness\b|\bothers\b|\bentertainment\b|\blifestyle\b|\bhealth\b', na=False)) & (
                                             ~final_article['ML_category'].str.contains(r'\bfood\b', na=False)), 'food',
                                         final_article['food_headline'])

        final_article['final_cat_1'] = final_article['final_cat_1'].str.replace('others', "")

        merged_columns = ['final_cat_1', 'final_cat_2', 'agriculture', 'oil_and_energy', 'aviation', 'blockchain',
                          'cybersecurity', 'consumer_tech', 'e_commerce', 'fintech', 'fmcg', 'hr', 'infrastructure',
                          'IT', 'SaaS', 'telecom', 'VC', 'logistics', 'startup', 'travel', 'food']

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
        category = 'Category: {}.'.format(all_categories)

    return render_template('home.html', prediction=category)


if __name__ == '__main__':
    app.run(debug=True)