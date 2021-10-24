import numpy as np
from copy import deepcopy

def apply_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

class WarmupLR(object):
    def __init__(self, init_lr, final_lr, steps):
        self.d=(final_lr-init_lr)/steps
        self.init_lr=init_lr
        self.cur_lr=init_lr
        self.index=0
        
    def step(self, optimizer):
        self.index+=1
        lr=self.index*self.d+self.init_lr
        apply_lr(optimizer, lr)
        self.cur_lr=lr

        return self.cur_lr

"""
decay by gamma after each milestone
"""
class EpochLR(object):
    def __init__(self, init_lr, milestones, gamma):
        self.cur_lr=init_lr
        self.steps=deepcopy(milestones)
        self.gamma=gamma
        self.index=0

    def resume(self, start_epoch):
        steps=deepcopy(self.steps)
        if steps[0]==0:
            steps.pop(0)
        
        i=0
        while i<len(steps) and start_epoch>=steps[i]:
            self.cur_lr*=self.gamma
            i+=1
        
        self.index=start_epoch
        self.update_steps()

        return self.get_cur_lr()

    def step(self, optimizer):
        raise NotImplementedError

    def update_steps(self):
        raise NotImplementedError
        
    def get_cur_lr(self):
        raise NotImplementedError

class StepLR(EpochLR):
    def __init__(self, init_lr, milestones, gamma):
        super(StepLR, self).__init__(init_lr, milestones, gamma)
    
    def update_steps(self):
        steps=deepcopy(self.steps)
        for p in steps:
            if p<=self.index:
                self.steps.pop(0)
                
    def get_cur_lr(self):
        return self.cur_lr
                
    def step(self, optimizer):
        self.index+=1
        if self.index in self.steps:
            self.cur_lr*=self.gamma
            self.steps.pop(0)
            if optimizer is not None:
                apply_lr(optimizer, self.cur_lr)

        return self.cur_lr

class LinearLR(EpochLR):
    def __init__(self, init_lr, milestones, gamma):
        super(LinearLR, self).__init__(init_lr, milestones, gamma)
        self.steps=[0]+self.steps
        self.cur_lr=init_lr
    
    def update_steps(self):
        steps=deepcopy(self.steps[1:])
        for p in steps:
            if p<=self.index:
                self.steps.pop(0)
                
    def get_cur_lr(self):
        if len(self.steps)==1:
            return self.cur_lr
        
        lr_next=self.cur_lr*self.gamma
        lr=(self.cur_lr-lr_next)*(self.steps[1]-self.index)/(self.steps[1]-self.steps[0])+lr_next
        
        return lr
                
    def step(self, optimizer):
        self.index+=1
        # if len(self.steps)==0 or self.index>self.steps[-1]:
        if len(self.steps)==1:
            return self.cur_lr
        
        cur_lr=self.get_cur_lr()

        if self.index in self.steps:
            self.cur_lr=cur_lr
            self.steps.pop(0)
        
        if optimizer is not None:
            apply_lr(optimizer, self.cur_lr)

        return cur_lr

class ExponentialLR(EpochLR):
    def __init__(self, init_lr, milestones, gamma):
        super(ExponentialLR, self).__init__(init_lr, milestones, gamma)
        self.exp_coeff=[]
        self.steps=[0]+self.steps
        for i in range(len(self.steps)-1):
            k=np.log(self.gamma)/(self.steps[i+1]-self.steps[i])
            self.exp_coeff.append(k)

    def update_steps(self):
        steps=deepcopy(self.steps[1:])
        for p in steps:
            if p<=self.index:
                self.steps.pop(0)
                self.exp_coeff.pop(0)
                
    def get_cur_lr(self):
        if len(self.exp_coeff)==0:
            return self.cur_lr

        cur_lr=self.cur_lr*np.exp(self.exp_coeff[0]*(self.index-self.steps[0]))
        
        return cur_lr

    def step(self, optimizer):
        self.index+=1

        if len(self.steps)==1:
            return self.cur_lr
        
        cur_lr=self.get_cur_lr()
        # if len(self.steps)>1 and self.index==self.steps[1]:
        if self.index in self.steps:
            self.cur_lr=cur_lr
            self.steps.pop(0)
            self.exp_coeff.pop(0)

        if optimizer is not None:
            apply_lr(optimizer, cur_lr)

        return cur_lr


class CosineLR(EpochLR):
    def __init__(self, init_lr, milestones, gamma):
        super(CosineLR, self).__init__(init_lr, milestones, gamma)
        self.cosine_coeff=[]
        self.steps=[0]+self.steps
        for i in range(len(self.steps)-1):
            k=np.arccos(self.gamma)/(self.steps[i+1]-self.steps[i])
            self.cosine_coeff.append(k)

    def update_steps(self):
        steps=deepcopy(self.steps[1:])
        for p in steps:
            if p<=self.index:
                self.steps.pop(0)
                self.cosine_coeff.pop(0)
                
    def get_cur_lr(self):
        if len(self.cosine_coeff)==0:
            return self.cur_lr
           
        cur_lr=self.cur_lr*np.cos(self.cosine_coeff[0]*(self.index-self.steps[0]))
        
        return cur_lr

    def step(self, optimizer):
        self.index+=1

        # if len(self.steps)==0 or self.index>self.steps[-1]:
        if len(self.steps)==1:
            return self.cur_lr
        
        cur_lr=self.get_cur_lr()
        if self.index in self.steps:
            self.cur_lr=cur_lr
            self.steps.pop(0)
            self.cosine_coeff.pop(0)

        if optimizer is not None:
            apply_lr(optimizer, cur_lr)
        
        return cur_lr

class CosineLRv2(EpochLR):
    def __init__(self, init_lr, milestones, gamma):
        super(CosineLRv2, self).__init__(init_lr, milestones, gamma)
        self.periods=[]
        self.scale=(1-gamma)*0.5
        self.min_val=(1+gamma)*0.5
        
        self.steps=[0]+self.steps
        for i in range(len(self.steps)-1):
            T=np.pi/(self.steps[i+1]-self.steps[i])
            self.periods.append(T)

    def update_steps(self):
        steps=deepcopy(self.steps[1:])
        for p in steps:
            if p<=self.index:
                self.steps.pop(0)
                self.periods.pop(0)

                
    def get_cur_lr(self):
        if len(self.periods)==0:
            return self.cur_lr

        scale=self.scale*np.cos(self.periods[0]*(self.index-self.steps[0]))+self.min_val
        cur_lr=self.cur_lr*scale
        
        return cur_lr

    def step(self, optimizer):
        self.index+=1

        if len(self.steps)==1:
            return self.cur_lr
        
        cur_lr=self.get_cur_lr()
        if self.index in self.steps:
            self.cur_lr=cur_lr
            self.steps.pop(0)
            self.periods.pop(0)

        if optimizer is not None:
            apply_lr(optimizer, cur_lr)
        
        return cur_lr

class CosineAnnealingLR(object):
    def __init__(self, max_lr, min_lr, T):
        assert max_lr>min_lr and isinstance(T, int)

        self.index=0
        self.max_lr=max_lr
        self.min_lr=min_lr

        self.k=0.5*(max_lr-min_lr)
        self.T=T

    def resume(self, start_epoch):
        self.index=start_epoch

        return self.get_cur_lr()

    def get_cur_lr(self):
        cur_lr=self.k*np.cos(2*np.pi/self.T*self.index)+self.k+self.min_lr

        return cur_lr

    def step(self, optimizer):
        self.index+=1

        cur_lr=self.get_cur_lr()

        if optimizer is not None:
            apply_lr(optimizer, cur_lr)
        
        return cur_lr

class DampeningLR(EpochLR):
    def __init__(self, init_lr, milestones, gamma, amp_ratio=0.1, num_periods=1, T=1):
        super(DampeningLR, self).__init__(init_lr, milestones, gamma)

        self.index=0
        self.amp_ratio=amp_ratio
        self.scale=(1-gamma)*0.5
        self.min_val=(1+gamma)*0.5
        
        self.steps=[0]+self.steps
        
        self.min_lr=init_lr
        for _ in range(len(milestones)):
            self.min_lr*=gamma

        self.num_periods=num_periods
        self.cur_T=1
        self.T=T

    def resume(self, start_epoch):
        self.index=start_epoch

        return self.get_cur_lr()

    def get_cos_lr(self, amp_decay=False):
        max_lr=self.cur_lr
        amp=max_lr*self.amp_ratio
        if amp_decay:
            if len(self.steps)>1:
                amp*=(self.steps[1]-self.index)/(self.steps[1]-self.steps[0])
        
        cos_lr=amp*0.5*np.cos(2*np.pi/self.cur_T*(self.index-self.steps[0]))

        return cos_lr+max_lr-amp*0.5
        
    def get_cur_lr(self):
        if len(self.steps)==1:
            return self.get_cos_lr(amp_decay=True)

        lr_next=self.cur_lr*self.gamma

        d=(self.cur_lr-lr_next)/(self.steps[1]-self.steps[0])*(self.index-self.steps[0])
        cos_lr=self.get_cos_lr(amp_decay=True)-d

        return cos_lr

    def update_steps(self):
        steps=deepcopy(self.steps[1:])
        for p in steps:
            if p<=self.index:
                self.steps.pop(0)

    def step(self, optimizer):
        self.index+=1
        if len(self.steps)>1:
            self.cur_T=(self.steps[1]-self.steps[0])/self.num_periods

        cur_lr=self.get_cur_lr()

        if self.index in self.steps:
            self.cur_lr=cur_lr
            self.steps.pop(0)
            if len(self.steps)==1:
                self.cur_T=self.T

        if optimizer is not None:
            apply_lr(optimizer, cur_lr)
        
        return cur_lr

def build_scheduler(scheduler_type, **kwargs):
    lr_scheduler=None
    if scheduler_type in ["cosineannealing", "dampening"]:
        if scheduler_type=="cosineannealing":
            lr_scheduler=CosineAnnealingLR(kwargs["init_lr"], kwargs["min_lr"], kwargs["T"])
        else:
            lr_scheduler=DampeningLR(kwargs["init_lr"], kwargs["milestones"], kwargs["gamma"], amp_ratio=kwargs["amp_ratio"], num_periods=kwargs["periods"], T=kwargs["T"])
    else:
        if scheduler_type=="step":
            scheduler=StepLR
        elif scheduler_type=="linear":
            scheduler=LinearLR
        elif scheduler_type=="exponential":
            scheduler=ExponentialLR
        elif scheduler_type=="cosine":
            scheduler=CosineLR
        elif scheduler_type=="cosinev2":
            scheduler=CosineLRv2

        else:
            raise RuntimeError("Unknown scheduler type: {}".format(scheduler_type))

        lr_scheduler=scheduler(kwargs["init_lr"], kwargs["milestones"], kwargs["gamma"])

    return lr_scheduler

def plot_lr():
    import matplotlib.pyplot as plt

    max_epochs=100
    milestones=[30,60,80]

    init_lr=0.1
    start=0
    gamma=0.5

    epochs=np.arange(start, max_epochs)

    sch1=LinearLR(init_lr, milestones, gamma)
    sch2=ExponentialLR(init_lr, milestones, gamma)
    sch3=CosineLR(init_lr, milestones, gamma)
    sch4=CosineLRv2(init_lr, milestones, gamma)
    sch5=CosineAnnealingLR(init_lr, init_lr*gamma, 10) 
    sch6=DampeningLR(init_lr, milestones, gamma, amp_ratio=0.1, num_periods=4, T=5) 

    schedulers=[sch1,sch2,sch3,sch4,sch5,sch6]
    for sch in schedulers:
        sch.resume(start)

    lr_values=np.zeros((6,max_epochs-start))

    for i in range(start, max_epochs):
        for j in range(6):
            lr_values[j,i-start]=schedulers[j].step(None)

    plt.figure(figsize=(10,8))
    plt.subplot(321)
    plt.title('Linear')
    plt.plot(epochs, lr_values[0])
    plt.subplot(322)
    plt.title('Exponential')
    plt.tight_layout()
    plt.plot(epochs, lr_values[1])
    plt.subplot(323)
    plt.title('Cosine')
    plt.tight_layout()
    plt.plot(epochs, lr_values[2])
    plt.subplot(324)
    plt.title('Cosinev2')
    plt.tight_layout()
    plt.plot(epochs, lr_values[3])
    plt.subplot(325)
    plt.title('CosineAnnealing')
    plt.tight_layout()
    plt.plot(epochs, lr_values[4])
    plt.subplot(326)
    plt.title('Dampening')
    plt.tight_layout()
    plt.plot(epochs, lr_values[5])

    # plt.show()
    plt.savefig("LR schedulers.png")



if __name__=='__main__':
    start=0
    scheduler=CosineLRv2(0.1, [20,30], 0.1)
    lr=scheduler.resume(start)
    print('Learn rate is {}'.format(lr))

    for i in range(start,40):
        '''step after train epoch'''
        lr=scheduler.step(None)
        print('Epoch: {}. Learn rate: {}'.format(i, lr))

    plot_lr()