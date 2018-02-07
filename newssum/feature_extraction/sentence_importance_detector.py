import subprocess
import os
from newssum.definitions import ROOT_DIR


class SentenceImportanceDetector():
    def __init__(self, sents, refers) -> None:
        self.sents = sents
        self.refers = refers

    def label_sents(self, out_dir):
        os.chdir(ROOT_DIR)
        print(os.getcwd())
        temp_file_path = "../output/temp.txt"
        JACANA_HOME = "../external/jacana-align"
        JAR_HOME = "../external/jacana-align/build/lib/jacana-align.jar"
        MODEL_HOME = "../external/jacana-align/scripts-align/Edingburgh_RTE2.all_sure.t2s.model"

        for i, sent in enumerate(self.sents):
            with open(temp_file_path, "w", encoding="utf8") as out:
                for ref in self.refers:
                    out.write(sent + "\t" + ref + "\n")

            output_file_path = out_dir + "/temp{}.json".format(i)
            drun = "java -DJACANA_HOME={} -jar {} -m {} -a {} -o {}".format(JACANA_HOME, JAR_HOME, MODEL_HOME,
                                                                            temp_file_path, output_file_path)
            subprocess.call(drun, shell=True)
