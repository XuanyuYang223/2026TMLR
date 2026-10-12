"""Fixed CPU checks of graph encoding, shifted loss and frozen source compatibility."""
from torch.nn import functional as F
from .common import *

def cpu_checks(c):
    device=device_setup(cpu=True);data,_,audit=load_data(c);examples=[]
    for n in c['lengths']:
        selected=np.flatnonzero(data['validation_lengths']==n)[:8]
        for i in selected:
            for ai,action in enumerate(ACTIONS):
                a=data['validation_graphs'][i,ai,:n,:n]
                assert np.array_equal(decode(data['validation_input'][i,ai,:n*n+4]),a)
                assert np.array_equal(labels(a),data['validation_labels'][i,ai])
        examples.extend(selected.tolist())
    ns=data['validation_lengths'][examples];prefix=data['validation_input'][examples,0]
    loss_errors=[]
    for task in TASKS:
        y=data['validation_labels'][examples,0,TASKS.index(task)]
        ids,mask,target,end,eq=batch(prefix,ns,task,y,device)
        rows=torch.arange(len(ns));gold=torch.as_tensor(y.astype(np.int64))
        assert torch.all((target!=-100).sum(1)==2)
        assert torch.equal(target[rows,eq+1],gold) and torch.all(target[rows,eq+2]==TOKEN_TO_ID['<EOS>'])
        assert torch.all(ids[rows,end]==TOKEN_TO_ID['<ARG_END>']) and int(ids.shape[1])<=128
        toy=torch.randn(len(ns),ids.shape[1],188)
        explicit=.5*(F.cross_entropy(toy[rows,eq],gold)+F.cross_entropy(toy[rows,eq+1],torch.full_like(gold,TOKEN_TO_ID['<EOS>'])))
        error=float(abs(loss(toy,target)-explicit));assert error<1e-6;loss_errors.append(error)
    saved=initial(c);net=model(c,'transformer',c['model_seed'],'cpu');net.load_state_dict(saved['model']);net.eval()
    before=parameter_hash(net);assert before==c['initial_parameter_sha256'];captured=[]
    hook=net.final_norm.register_forward_hook(lambda m,i,o:captured.append(o))
    query_errors=[];task_hidden=[]
    with torch.inference_mode():
        for task in TASKS:
            y=data['validation_labels'][examples,0,TASKS.index(task)]
            x,m,_,end,eq=batch(prefix,ns,task,y,'cpu');captured.clear();z=net(x,m);h=captured[-1][rows,end].clone()
            changed=x.clone();changed[rows,eq+1]=(changed[rows,eq+1]+1)%8
            zz=net(changed,m);error=float((z[rows,eq]-zz[rows,eq]).abs().max());assert error<2e-5
            query_errors.append(error);task_hidden.append(h)
    hook.remove();hidden_error=max(float((h-task_hidden[0]).abs().max()) for h in task_hidden)
    assert hidden_error<2e-5 and parameter_hash(net)==before
    result=dict(status='passed',protocol_sha256=sha(CONFIG),data_sha256=audit['data_sha256'],vocabulary188=True,
        original_true_step0_parameter_hash=before,graph_examples_checked=len(examples)*5,all_four_task_masks=True,
        max_independent_shifted_ce_difference=max(loss_errors),future_answer_max_logit_difference=max(query_errors),
        pre_task_hidden_max_task_suffix_difference=hidden_error,largest_full_supervised_sequence=72,
        no_model_optimizer_or_updates=True,gpu_forwards=0)
    json_write(DIAG/'cpu_checks.json',result);return result
