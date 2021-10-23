import wandb


class EarlyStopping(object):
    def __init__(self, early_stop):
        self.early_stop = early_stop
        self.best_recall = 0
        self.best_epoch = 0
        self.str = ""
        self.desc_times = 0

    def check_stop(self, evaluators, epoch):
        recall = evaluators["valid"].get("Recall", 20)
        # wandb.log({"val/Recall@20": recall, "epoch": epoch})
        if recall > self.best_recall:
            self.desc_times = 0
            self.best_recall = recall
            self.best_epoch = epoch
            if "test" in evaluators:
                # wandb.run.summary.update(
                #     {
                #         "test/Recall@20": evaluators["test"].get("Recall", 20),
                #         "test/NDCG@20": evaluators["test"].get("NDCG", 20),
                #         "test/ARP@20": evaluators["test"].get("ARP", 20),
                #         "best epoch": epoch,
                #     }
                # )
                self.str = evaluators["test"].__str__()
            else:
                self.str = evaluators["valid"].__str__()
        else:
            self.desc_times += 1
            if self.early_stop is not None and self.desc_times == self.early_stop:
                print("Early stopping triggered.")
                return True
        return False

    def __str__(self):
        return "Best epoch: {}.\n{}".format(self.best_epoch, self.str)
