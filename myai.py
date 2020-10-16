import mytorch

class Learner:
    def __init__(self,trdl,model,opt,loss_fn,valdl=None,cbs=None):
        self.trdl,self.valdl,self.model=trdl,valdl,model
        self.opt,self.loss_fn=opt,loss_fn
        self.cbs=[] if cbs is None else cbs
        for cb in cbs:
            cb.learn=self

    def call_cbs(self,name):
        for cb in self.cbs:
            meth=getattr(cb,name,None)
            if meth: meth()

    def fit(self,n_epochs):
        self.call_cbs('before_fit')
        for ep in range(n_epochs):
            self.epoch=ep
            self.training=True
            self.call_cbs('before_epoch')
            for self.xb,self.yb in self.trdl:
                self.preds=self.model(self.xb)
                self.loss=self.loss_fn(self.preds,self.yb)
                self.call_cbs('after_loss')
                self.loss.backward()
                self.opt.step()

            self.training=False
            with mytorch.no_grad():
                for self.xb,self.yb in self.valdl:
                    self.preds=self.model(self.xb)
                    self.loss=self.loss_fn(self.preds,self.yb)
                    self.call_cbs('after_loss')
            self.call_cbs('after_epoch')
        self.call_cbs('after_fit')
