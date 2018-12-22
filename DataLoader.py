import xml.etree.ElementTree as ElementTree
from collections import defaultdict

default_filepath = {
    'SemEval2017/data/train/SemEval2016-Task3-CQA-QL-train-part1.xml':0,
    'SemEval2017/data/train/SemEval2016-Task3-CQA-QL-train-part2.xml':0,
    'SemEval2017/data/scorer_v2.3/SemEval2017-Task3-CQA-QL-dev.xml':1,
    'SemEval2017/data/train/SemEval2016-Task3-CQA-QL-test.xml':0,
    'SemEval2017/data/test/SemEval2017-task3-English-test.xml':2
  }

LABELS = {
  "PerfectMatch": 1,
  "Relevant" : 0.7,
  "Irrelevant": 0
  }

def load():  
  questions = load_questions(default_filepath)
  return questions

def parseTask3TrainingData(filepath):
  tree = ElementTree.parse(filepath)
  root = tree.getroot()
  OrgQuestions = {}
  for OrgQuestion in root.iter('OrgQuestion'):
    OrgQuestionOutput = {}
    OrgQuestionOutput['id'] = OrgQuestion.attrib['ORGQ_ID']
    OrgQuestionOutput['subject'] = OrgQuestion.find('OrgQSubject').text
    OrgQuestionOutput['question'] = OrgQuestion.find('OrgQBody').text
    OrgQuestionOutput['comments'] = {}
    OrgQuestionOutput['related'] = {}
    OrgQuestionOutput['featureVector'] = []
    if OrgQuestionOutput['id'] not in OrgQuestions:
      OrgQuestions[OrgQuestionOutput['id']] = OrgQuestionOutput
    for RelQuestion in OrgQuestion.iter('RelQuestion'):
      RelQuestionOutput = {}
      RelQuestionOutput['id'] = RelQuestion.attrib['RELQ_ID']
      RelQuestionOutput['subject'] = RelQuestion.find('RelQSubject').text
      RelQuestionOutput['question'] = RelQuestion.find('RelQBody').text
      RelQuestionOutput['givenRelevance'] = RelQuestion.attrib['RELQ_RELEVANCE2ORGQ']
      RelQuestionOutput['givenRank'] = RelQuestion.attrib['RELQ_RANKING_ORDER']
      RelQuestionOutput['comments'] = {}
      RelQuestionOutput['featureVector'] = []
      for RelComment in OrgQuestion.iter('RelComment'):
        RelCommentOutput = {}
        RelCommentOutput['id'] = RelComment.attrib['RELC_ID']
        RelCommentOutput['date'] = RelComment.attrib['RELC_DATE']
        RelCommentOutput['username'] = RelComment.attrib['RELC_USERNAME']
        RelCommentOutput['comment'] = RelComment.find('RelCText').text
        RelQuestionOutput['comments'][RelCommentOutput['id']] = RelCommentOutput
        #if RelQuestionOutput['question'] != None:
        if RelQuestionOutput['question'] == None:
          RelQuestionOutput['question'] = ""
      OrgQuestions[OrgQuestionOutput['id']]['related'][RelQuestionOutput['id']] = RelQuestionOutput
      #else:
      #print("Warning: skipping empty question " + RelQuestionOutput['id'])
  return OrgQuestions

def load_questions(filenames):
  output = {}
  for filePath in filenames:
    print("\nParsing %s" % filePath)
    fileoutput = parseTask3TrainingData(filePath)
    print("  Got %s primary questions" % len(fileoutput))
    if not len(fileoutput):
      raise Exception("Failed to load any entries from " + filePath)
    isTraining = default_filepath[filePath]
    for q in fileoutput:
      fileoutput[q]['isTraining'] = isTraining
    output.update(fileoutput)
  print("\nTotal of %s entries" % len(output))
  return output

def make_question_pair():
  questions_dict = load()
  train_set, dev_set, test_set = [], [], []
  for question in questions_dict:
    orig_question = questions_dict[question]['question']
    for rel_question_id in questions_dict[question]['related']:
      rel_question = questions_dict[question]['related'][rel_question_id]['question']
      #label = 0 if questions_dict[question]['related'][rel_question_id]['givenRelevance'] == 'Irrelevant' else 1
      label = LABELS[questions_dict[question]['related'][rel_question_id]['givenRelevance']]
      if questions_dict[question]['isTraining'] == 0:
        train_set.append([orig_question, rel_question, label])
      elif questions_dict[question]['isTraining'] == 1:
        dev_set.append([orig_question, rel_question, label])
      else:
        test_set.append([orig_question, rel_question, label])
  return train_set, dev_set, test_set

def get_test_set():
  test_path = "SemEval2017/data/test/SemEval2017-task3-English-test-input.xml"
  sent_pair_dict = defaultdict(list)
  all_question = {}
  questions_dict = parseTask3TrainingData(test_path)
  for question in questions_dict:
    all_question[question] = questions_dict[question]['question']
    for rel_question_id in questions_dict[question]['related']:
      rel_question = questions_dict[question]['related'][rel_question_id]['question']
      all_question[rel_question_id] = rel_question
      sent_pair_dict[question].append(rel_question_id)
  return sent_pair_dict, all_question
