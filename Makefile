# ----------------------------------------------------------------------
# Parameters

FEAT = bert
BERT = mbert
CONFIG = config.ini

GPU = 0

#BUCKETS = --buckets=48
#BATCH_SIZE = --batch-size=500
#MAX_SENT_LENGTH=--max-sent-length 140
#ATTN=--attention-layer=6

#----------------------------------------------------------------------
# Corpora

CORPUS_DIR = ../train-dev
CORPUS_TRAIN = $(CORPUS_DIR)/UD_$(RES2)/$(CORPUS)-ud-train.conllu
CORPUS_DEV = $(CORPUS_DIR)/UD_$(RES2)/$(CORPUS)-ud-dev.conllu
CORPUS_TEST = $(CORPUS_DIR)/UD_$(RES2)/$(CORPUS)-ud-test.conllu

#BLIND_TEST=$(CORPUS_DIR)/../test-udpipe/$(LANG).conllu
#BLIND_TEST=$(CORPUS_DIR)/../test-stanza-sent/$(LANG).conllu
#BLIND_TEST=$(CORPUS_DIR)/../EDparser/data/iwpt2020/test-udpipe/$(LANG).conllu
BLIND_TEST=$(CORPUS_DIR)/../iwpt2020stdata/sysoutputs/turkunlp/primary/$(LANG).conllu

GOLD_TEST= $(CORPUS_DIR)/../iwpt2020stdata/$(UD_TOOLS)/../test-gold/$(LANG).conllu

UD_TOOLS = ../iwpt2020stdata/tools
EVALB = python ../eval.py
EVAL07 = perl ../eval07.pl

ifeq ($(LANG), ar)
  CORPUS=ar_padt
  RES2=Arabic-PADT
  MODEL = --bert=asafaya/bert-large-arabic #TurkuNLP/wikibert-base-ar-cased
  BERT = asafaya
else ifeq ($(LANG), ba)
  CORPUS=ba
  RES2=Baltic
  #MODEL = --bert=TurkuNLP/wikibert-base-lv-cased
else ifeq ($(LANG), bg)
  CORPUS=bg_btb
  RES2=Bulgarian-BTB
  MODEL = --bert=DeepPavlov/bert-base-bg-cs-pl-ru-cased #TurkuNLP/wikibert-base-bg-cased #iarfmoose/roberta-base-bulgarian
  BERT = DeepPavlov
else ifeq ($(LANG), ca)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=ca_ancora
  RES2=Catalan-AnCora
  #MODEL= --bert=TurkuNLP/wikibert-base-ca-cased
  bert = mbert
else ifeq ($(LANG), cs) #dev PDT
  CORPUS=cs_pdt
  RES2=Czech-PDT
  MODEL = --bert=DeepPavlov/bert-base-bg-cs-pl-ru-cased
  BERT = DeepPavlov
else ifeq ($(LANG), de)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=de_hdt
  RES2=German-HDT
  MODEL = --bert=dbmdz/bert-base-german-uncased
  BERT = dbmdz-bert-base
else ifeq ($(LANG), en)
  CORPUS=en_ewt
  RES2=English-EWT
  MODEL = --bert=google/electra-base-discriminator
  BERT = electra-base
else ifeq ($(LANG), ptb)
 CORPUS_DIR=..
  CORPUS=en_ptb
  CORPUS_TRAIN = $(CORPUS_DIR)/SD_English_PTB/$(CORPUS)-sd-train.conllu
  CORPUS_DEV = $(CORPUS_DIR)/SD_English_PTB/$(CORPUS)-sd-dev.conllu
  BLIND_TEST = $(CORPUS_DIR)/SD_English_PTB/$(CORPUS)-sd-test.conllu
  GOLD_TEST = $(CORPUS_DIR)/SD_English_PTB/en_ptb-sd-test.conllu
  MODEL = --bert=google/electra-base-discriminator
  BERT = electra-base
else ifeq ($(LANG), es)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=es_ancora
  RES2=Spanish-AnCora
  #MODEL = --bert=skimai/electra-small-spanish # TurkuNLP/wikibert-base-es-cased
  bert = mbert
else ifeq ($(LANG), et) #dev EDT
  CORPUS=et_edt_ewt
  RES2=Estonian-EDT-EWT
  #MODEL = --bert=TurkuNLP/wikibert-base-et-cased
  bert = mbert
else ifeq ($(LANG), fi)
  CORPUS=fi_tdt
  RES2=Finnish-TDT
  MODEL = --bert=TurkuNLP/bert-base-finnish-cased-v1
  #MODEL = --bert=TurkuNLP/wikibert-base-fi-cased
  BERT = TurkuNLP
  #ATTN=--attention-layer=8
else ifeq ($(LANG), fr)
  CORPUS=fr_sequoia
  RES2=French-Sequoia
  MODEL = --bert=camembert/camembert-large #camembert-base TurkuNLP/wikibert-base-fr-cased
  BERT = camembert-large
else ifeq ($(LANG), it)
  CORPUS=it_isdt
  RES2=Italian-ISDT
  MODEL = --bert=dbmdz/electra-base-italian-xxl-cased-discriminator
  BERT = dbmdz-electra-xxl
else ifeq ($(LANG), ja)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=ja_gsd
  RES2=Japanese-GSD
  MODEL = --bert=cl-tohoku/bert-base-japanese
  bert = cl-tohoku-bert
else ifeq ($(LANG), la)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=la_ittb_llct
  RES2=Latin-ITTB-LLCT
  #MODEL = --bert=ponteineptique/latin-classical-small
  BERT = mbert
else ifeq ($(LANG), lt)
  CORPUS=lt_alksnis
  RES2=Lithuanian-ALKSNIS
  #MODEL = --bert=TurkuNLP/wikibert-base-lt-cased
  ATTN=--attention-layer=8
else ifeq ($(LANG), lv)
  CORPUS=lv_lvtb
  RES2=Latvian-LVTB
  #MODEL = --bert=TurkuNLP/wikibert-base-lv-cased
else ifeq ($(LANG), nl) #dev Alpino
  CORPUS=nl_alpino
  RES2=Dutch
  #MODEL = --bert=TurkuNLP/wikibert-base-nl-cased
  MODEL = --bert=wietsedv/bert-base-dutch-cased
  BERT = wietsedv
else ifeq ($(LANG), no)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=no_nynorsk
  RES2=Norwegian-Nynorsk
  MODEL = --bert=TurkuNLP/wikibert-base-no-cased
  bert = mbert
else ifeq ($(LANG), pl) #dev LFG
  CORPUS=pl_pdb_lfg
  RES2=Polish-PDB-LFG
  MODEL = --bert=dkleczek/bert-base-polish-cased-v1 #DeepPavlov/bert-base-bg-cs-pl-ru-cased
  BERT = dkleczek
else ifeq ($(LANG), ro)
  CORPUS_DIR=../ud-treebanks-v2.6
  BLIND_TEST= $(CORPUS_TEST)
  GOLD_TEST= $(CORPUS_TEST)
  CORPUS=ro_rrt
  RES2=Romanian-RRT
  bert = mbert
else ifeq ($(LANG), ru)
  CORPUS=ru_syntagrus
  RES2=Russian-SynTagRus
  MODEL = --bert=DeepPavlov/rubert-base-cased
  BERT = DeepPavlov
else ifeq ($(LANG), sk)
  CORPUS=sk_snk
  RES2=Slovak-SNK
  #MODEL = --bert=TurkuNLP/wikibert-base-sk-cased
else ifeq ($(LANG), sv)
  CORPUS=sv_talbanken
  RES2=Swedish-Talbanken
  MODEL = --bert=KB/bert-base-swedish-cased
  BERT = KB
else ifeq ($(LANG), ta)
  CORPUS=ta_ttb
  RES2=Tamil-TTB
  BLIND_TEST = $(CORPUS_DIR)/../test-udpipe/$(LANG).conllu
  #MODEL = --bert=monsoon-nlp/tamillion
else ifeq ($(LANG), uk)
  CORPUS=uk_iu
  RES2=Ukrainian-IU
  MODEL = --bert=TurkuNLP/wikibert-base-uk-cased
  BERT = TurkuNLP
  # nu=0.9
else ifeq ($(LANG), zh)
  CORPUS=zh_ctb7
  CORPUS_TRAIN = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-train.conllu
  CORPUS_DEV = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-dev.conllu
  BLIND_TEST = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-test.conllu
  GOLD_TEST = $(CORPUS_DIR)/CoNLL09/$(CORPUS)-test.conllu
  MODEL = --bert=hfl/chinese-electra-base-discriminator # bert-base-chinese # hfl/chinese-electra-large-discriminator
  BERT = hfl-eletcra-base
else
  CORPUS_TRAIN= data/CoNLL2009-ST-English-train.conll
  CORPUS_DEV  = data/CoNLL2009-ST-English-development.conll
  CORPUS_TEST = data/CoNLL2009-ST-English-test-wsj.conll
endif

#----------------------------------------------------------------------
# Targets

.PRECIOUS: exp/$(CORPUS).$(BERT)$(VER)/model

#BERT =dbmdz-electra-xxl

TARGET=exp/$(CORPUS).$(BERT)$(VER)

# relate LANG to CORPUS
exp/$(LANG)$(VER)%: $(TARGET)%
	@:

$(TARGET)/model:
	python -u -m diaparser.cmds.biaffine_dependency train -d=$(GPU) -p=$@ \
	   -c=$(CONFIG) $(MODEL) $(ATTN) \
	   --train=$(CORPUS_TRAIN) $(MAX_SENT_LENGTH) $(BATCH_SIZE) $(BUCKETS) \
	   --dev=$(CORPUS_DEV) --feat=$(FEAT)

$(TARGET).test.conllu: $(TARGET)/model
	python -m diaparser.cmds.biaffine_dependency predict -d=$(GPU) -p=$< --tree \
	   --data=$(BLIND_TEST) \
	   --pred=$@
	python $(CORPUS_DIR)/../fix-root.py $@

LANGS=ar bg cs en et fi fr it lt lv nl pl ru sk sv ta uk 
LANGS1=ar bg en et fi sk
LANGS2=fr it ru ta uk sv
LANGS3=lv lt nl pl cs

UD_LANGS=ca de es ja no ro

all:
	for l in $(LANGS); do \
	    $(MAKE) -s GPU=$(GPU) LANG=$$l FEAT=$(FEAT) VER=$(VER) exp/$${l}$(VER).test.eval &>> exp/$${l}$(VER).test.make; \
	done

all-ud:
	for l in $(UD_LANGS); do \
	    $(MAKE) -s GPU=$(GPU) LANG=$$l FEAT=$(FEAT) VER=$(VER) exp/$${l}$(VER).test.eval07 &>> exp/$${l}$(VER).test.make; \
	done

train:
	for l in $(LANGS); do \
	    nohup ${MAKE} -s GPU=$(GPU) LANG=$$l exp/$$l$(VER)/model &>> exp/$${l}$(VER).train.make; \
	done

# ----------------------------------------------------------------------
# Evaluation

$(TARGET).test.eval: $(TARGET).test.conllu
	perl $(UD_TOOLS)/enhanced_collapse_empty_nodes.pl $< > $(TARGET).test.nen.conllu
	python $(UD_TOOLS)/iwpt20_xud_eval.py -v $(UD_TOOLS)/../test-gold/$(LANG).nen.conllu $(TARGET).test.nen.conllu > $@

$(TARGET).test.evalb: $(TARGET).test.eval
	$(EVALB) -g $(GOLD_TEST) -s $@ --evalb

$(TARGET).test.eval07: $(TARGET).test.conllu
	$(EVAL07) -p -q -g $(GOLD_TEST) -s $< > $@

evaluate:
	for l in $(LANGS); do \
	   $(MAKE) -s GPU=$(GPU) LANG=$$l exp/$$l.$(BERT).test.evalb &>> exp/$$l.$(BERT).test.make; \
	done

exp/test.eval: evaluate
	( cd exp; python ../eval-summary.py > $(notdir $@) )

baltic:
	for l in et lt lv; do \
	  python -m diaparser.cmds.biaffine_dependency predict -d=$(GPU) --feat=$(FEAT) \
	   -p=exp/ba.$(BERT) --tree \
	   $(subst $(LANG),$$l,$(BLIND_TEST)) \
	   --pred=exp/$$l.$(BERT)-ba.test.conllu; \
	done

# ----------------------------------------------------------------------
# Run tests

lint:
	flake8 diaparser --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

test:
	pytest -s tests

# ----------------------------------------------------------------------
# Parse plain text

TEXT_FILE=

# example
# make GPU=2 LAN=it CORPUS_DIR=/project/piqasso/Collection/IWPT20 TEXT_FILE=/project/piqasso/Collection/IWPT20/train-dev/UD_Italian-ISDT/it_isdt-ud-dev.txt exp/it-bert-raw-text.conllu
exp/$(LAN).$(BERT)$(VER)-$(TEXT_FILE).conllu: exp/$(LAN).$(BERT)$(VER)/model
	python -m diaparser.cmds.biaffine_dependency predict -d=$(GPU) -p=$< --tree --text $(LAN) \
	   $(TEXT_FILE) \
	   --pred=$@
