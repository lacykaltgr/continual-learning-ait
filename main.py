import wandb
import numpy as np
import os
import tensorflow as tf

import mir
from utils import get_logger, get_temp_logger, logging_per_task, naive_cross_entropy_loss
from copy import deepcopy
from pydoc import locate
from classifier import ResNet18, classifier
from loss import calculate_loss
from data_preparation import CLDataLoader
from stable_diffusion.stable_diffusion import StableDiffusion

# Obligatory overhead
# -----------------------------------------------------------------------------------------

result_dir = 'results'

result_path = os.path.join('Results', result_dir)
if not os.path.exists(result_path): os.mkdir(result_path)
sample_path = os.path.join(*['Results', result_dir, 'samples/'])
if not os.path.exists(sample_path): os.mkdir(sample_path)
recon_path = os.path.join(*['Results', result_dir, 'reconstructions/'])
if not os.path.exists(recon_path): os.mkdir(recon_path)
mir_path = os.path.join(*['Results', result_dir, 'mir/'])
if not os.path.exists(mir_path): os.mkdir(mir_path)


device = 'cuda:0' if tf.test.is_gpu_available() else 'cpu'
n_runs = 1
n_tasks = 10
n_epochs = 100
n_classes = 10
input_size = 100
samples_per_task = 100
batch_size = 100
use_conv = True
gen_depth = 6
cls_mir_gen = 1
cls_iters = 100
gen_mir_gen = 1
gen_iters = 10
lr = 0.001
warmup = 0
max_beta = 1
reuse_samples = True
print_every = True
mem_coeff = 0.12

# fetch data
data = locate('data.get_%s')()


# make dataloader
train_loader, val_loader, test_loader = \
    [CLDataLoader(elem, None, train=t) for elem, t in zip(data, [True, False, False])]

# create logging containers
LOG = get_logger(['gen_loss', 'cls_loss', 'acc'], n_runs=n_runs, n_tasks=n_tasks)



# Train the model
# -----------------------------------------------------------------------------------------

# --------------
# Begin Run Loop
for run in range(n_runs):

    mir_tries, mir_success = 0, 0

    # CLASSIFIER
    if use_conv:
        cls = ResNet18(n_classes, nf=20, input_size= input_size)
    else:
        cls = classifier().to(device)

    opt = tf.keras.optimizers.SGD(cls.parameters(), lr=lr)
    if run == 0:
        print("number of classifier parameters:", sum([np.prod(p.size()) for p in cls.parameters()]))

    # GENERATIVE MODELING
    gen = StableDiffusion().to(device)
    opt_gen = tf.keras.optimizers.Adam(gen.parameters())
    if run == 0:
        print("number of generator parameters: ", sum([np.prod(p.size()) for p in gen.parameters()]))
    # INIT
    prev_gen, prev_model = None, None


    #----------------
    # Begin Task Loop
    for task, tr_loader in enumerate(train_loader):

        print('\n--------------------------------------')
        print(f'Run #{run} Task #{task} TRAIN')
        print('--------------------------------------\n')

        cls = cls.train()
        gen = gen.train()

        sample_amt = 0

        # ----------------
        # Begin Epoch Loop
        for epoch in range(n_epochs):

            #---------------------
            # Begin Minibatch Loop
            for i, (data, target) in enumerate(tr_loader):

                if sample_amt > samples_per_task > 0: break
                sample_amt += data.size(0)

                data, target = data.to(device), target.to(device)

                beta = min([(sample_amt) / max([warmup, 1.]), max_beta])

                #------ Train Generator ------#

                #-------------------------------
                # Begin Generator Iteration Loop
                for it in range(gen_iters):

                    x_mean, z_mu, z_var, ldj, z0, zk = gen(data)
                    gen_loss, rec, kl, _ = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

                    tot_gen_loss = 0 + gen_loss

                    if task > 0:

                        if it == 0 or not reuse_samples:
                            mem_x, mir_worked = mir.retrieve_gen_for_gen(args, data, gen, prev_gen, prev_cls)

                            mir_tries += 1
                            if mir_worked:
                                mir_success += 1
                                # keep for logging later
                                gen_x, gen_mem_x = data, mem_x

                        mem_x_mean, z_mu, z_var, ldj, z0, zk = gen(mem_x)
                        mem_gen_loss, mem_rec, mem_kl, _ = calculate_loss(mem_x_mean, mem_x, z_mu, z_var, z0, zk, ldj, args, beta=beta)

                        tot_gen_loss += mem_coeff * mem_gen_loss

                    opt_gen.zero_grad()
                    tot_gen_loss.backward()
                    opt_gen.step()

                # End Generator Iteration Loop
                #------------------------------

                if gen is not None:
                    if i % print_every == 0:
                        print(f'current VAE loss = {gen_loss.item():.4f} (rec: {rec.item():.4f} + beta: {beta:.2f} * kl: {kl.item():.2f}')
                        if task > 0:
                            print(f'memory VAE loss = {mem_gen_loss.item():.4f} (rec: { mem_rec.item():.4f} + beta: {beta:.2f} * kl: {mem_kl.item():.2f})')


                #------ Train Classifier-------#

                #--------------------------------
                # Begin Classifier Iteration Loop
                for it in range(cls_iters):

                    logits = cls(data)
                    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='mean')
                    cls_loss = loss_fn(target, logits)
                    tot_cls_loss = 0 + cls_loss

                    if task > 0:

                        if it == 0 or not reuse_samples:
                            mem_x, mem_y, mir_worked = mir.retrieve_gen_for_cls(args, data, cls, prev_cls, prev_gen)
                            mir_tries += 1
                            if mir_worked:
                                mir_success += 1
                                # keep for logging later
                                cls_x, cls_mem_x = data, mem_x

                        mem_logits = cls(mem_x)

                        mem_cls_loss = naive_cross_entropy_loss(mem_logits, mem_y)

                        tot_cls_loss += mem_coeff * mem_cls_loss

                    opt.zero_grad()
                    tot_cls_loss.backward()
                    opt.step()

                # End Classifer Iteration Loop
                #-----------------------------

                if i % print_every == 0:
                    pred = logits.argmax(dim=1, keepdim=True)
                    acc = pred.eq(target.view_as(pred)).sum().item() / pred.size(0)
                    print(f'current training accuracy: {acc:.2f}')
                    if task > 0:
                        pred = mem_logits.argmax(dim=1, keepdim=True)
                        mem_y = mem_y.argmax(dim=1, keepdim=True)
                        acc = pred.eq(mem_y.view_as(pred)).sum().item() / pred.size(0)
                        print(f'memory training accuracy: {acc:.2f}')

            # End Minibatch Loop
            #-------------------

        # End Epoch Loop
        #---------------

        # ------------------------ eval ------------------------ #

        print('\n--------------------------------------')
        print('Run #{} Task #{} EVAL'.format(run, task))
        print('--------------------------------------\n')

        with tf.GradientTape(persistent=True) as tape:

            cls = cls.eval()
            prev_cls = deepcopy(cls)

            gen = gen.eval()
            prev_gen = deepcopy(gen)

            # save some training reconstructions:
            recon_path_ = os.path.join(recon_path, f'task{task}.png')
            recons = tf.concat([data.to('cpu'), x_mean.to('cpu')])
            save_image(recons, recon_path_, nrow=batch_size)

            # save some pretty images:
            gen_images = gen.generate(25).to('cpu')
            sample_path_ = os.path.join(sample_path,'task{}.png'.format(task))
            save_image(gen_images, sample_path_, nrow=5)


            # save some MIR samples:
            if task > 0:
                mir_images = tf.concat([cls_x.to('cpu'), cls_mem_x.to('cpu')])
                mir_path_ = os.path.join(mir_path, f'cls_task{task}.png')
                save_image(mir_images, mir_path_, nrow=10)

                mir_images = tf.concat([gen_x.to('cpu'), gen_mem_x.to('cpu')])
                mir_path_ = os.path.join(mir_path, f'gen_task{task}.png')
                save_image(mir_images, mir_path_, nrow=10)

            eval_loaders = [('valid', val_loader), ('test', test_loader)]


            #----------------
            # Begin Eval Loop
            for mode, loader_ in eval_loaders:

                #----------------
                # Begin Task Eval Loop
                for task_t, te_loader in enumerate(loader_):
                    if task_t > task: break
                    LOG_temp = get_temp_logger(None, ['gen_loss', 'cls_loss', 'acc'])

                    #---------------------
                    # Begin Minibatch Eval Loop
                    for i, (data, target) in enumerate(te_loader):
                        #if args.cuda:
                        data, target = data.to(device), target.to(device)

                        logits = cls(data)

                        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='mean')(target, logits)
                        pred = logits.argmax(dim=1, keepdim=True)

                        LOG_temp['acc'] += [pred.eq(target.view_as(pred)).sum().item() / pred.size(0)]
                        LOG_temp['cls_loss'] += [loss.item()]

                        if gen is not None:
                            x_mean, z_mu, z_var, ldj, z0, zk = gen(data)
                            gen_loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0,
                                    zk, ldj, args, beta=beta)
                            LOG_temp['gen_loss'] += [gen_loss.item()]

                    # End Minibatch Eval Loop
                    #-------------------

                    logging_per_task(wandb, LOG, run, mode, 'acc', task, task_t,
                             np.round(np.mean(LOG_temp['acc']), 2))
                    logging_per_task(wandb, LOG, run, mode, 'cls_loss', task, task_t,
                             np.round(np.mean(LOG_temp['cls_loss']), 2))
                    logging_per_task(wandb, LOG, run, mode, 'gen_loss', task, task_t,
                             np.round(np.mean(LOG_temp['gen_loss']), 2))


                # End Task Eval Loop
                #-------------------

                print(f'\n{mode}:')
                print(LOG[run][mode]['acc'])

            # End Eval Loop
            #--------------

        # End torch.no_grad()
        #--------------------

    # End Task Loop
    #--------------

    print('--------------------------------------')
    print(f'Run #{run} Final Results')
    print('--------------------------------------')
    for mode in ['valid', 'test']:

        # accuracy
        final_accs = LOG[run][mode]['acc'][:, task]
        logging_per_task(wandb, LOG, run, mode, 'final_acc', task, value=np.round(np.mean(final_accs),2))

        # forgetting
        best_acc = np.max(LOG[run][mode]['acc'], 1)
        final_forgets = best_acc - LOG[run][mode]['acc'][:, task]
        logging_per_task(wandb, LOG, run, mode, 'final_forget', task, value=np.round(np.mean(final_forgets[:-1]),2))

        # VAE loss
        final_elbos = LOG[run][mode]['gen_loss'][:, task]
        logging_per_task(wandb, LOG, run, mode, 'final_elbo', task, value=np.round(np.mean(final_elbos), 2))

        print(f'\n{mode}:')
        print(f'final accuracy: {final_accs}')
        print(f'average: {LOG[run][mode]["final_acc"]}')
        print(f'final forgetting: {final_forgets}')
        print(f'average: {LOG[run][mode]["final_forget"]}')
        print(f'final VAE loss: {final_elbos}')
        print(f'average: {LOG[run][mode]["final_elbo"]}\n')

        try:
            mir_worked_frac = mir_success/ (mir_tries)
            logging_per_task(wandb, LOG, run, mode, 'final_mir_worked_frac', task, mir_worked_frac)
            print('mir worked \n', mir_worked_frac)
        except:
            pass

# End Run Loop
#-------------

print('--------------------------------------')
print('--------------------------------------')
print('FINAL Results')
print('--------------------------------------')
print('--------------------------------------')
for mode in ['valid','test']:

    # accuracy
    final_accs = [LOG[x][mode]['final_acc'] for x in range(n_runs)]
    final_acc_avg = np.mean(final_accs)
    final_acc_se = np.std(final_accs) / np.sqrt(n_runs)

    # forgetting
    final_forgets = [LOG[x][mode]['final_forget'] for x in range(n_runs)]
    final_forget_avg = np.mean(final_forgets)
    final_forget_se = np.std(final_forgets) / np.sqrt(n_runs)

    # VAE loss
    final_elbos = [LOG[x][mode]['final_elbo'] for x in range(n_runs)]
    final_elbo_avg = np.mean(final_elbos)
    final_elbo_se = np.std(final_elbos) / np.sqrt(n_runs)

    # MIR worked
    try:
        final_mir_worked_frac = [LOG[x][mode]['final_mir_worked_frac'] for x in range(args.n_runs)]
        final_mir_worked_avg = np.mean(final_mir_worked_frac)
    except:
        pass

    print('\nFinal {} Accuracy: {:.3f} +/- {:.3f}'.format(mode, final_acc_avg, final_acc_se))
    print('\nFinal {} Forget: {:.3f} +/- {:.3f}'.format(mode, final_forget_avg, final_forget_se))
    print('\nFinal {} ELBO: {:.3f} +/- {:.3f}'.format(mode, final_elbo_avg, final_elbo_se))