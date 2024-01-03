from argparse import ArgumentParser
from time import sleep
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import (
    GPT2TokenizerFast,
    GPT2Tokenizer
)
from torch.optim import Adam

from datetime import datetime
from .data.knowledge import KnowledgeDataset
import numpy as np
import openai
import os
import asyncio
import time
import sys
import re
import json
import copy
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
FACTOR = 10000000

class ExampleSelection(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--dirpath", type=str, default="models"
        )
        parser.add_argument(
            "--knowledge_data_path",
            type=str,
            default="datasets/gsm8k_/code-davinci-002/gsm8k_pool_100.json",
        )
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="datasets/gsm8k_/gsm8k_train_100.json",
        )
        parser.add_argument(
            "--valid_data_path",
            type=str,
            default="datasets/gsm8k_/gsm8k_val_100.json",
        )
        parser.add_argument("--batch_size", type=int, default=10)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--num_workers", type=int, default=0)

        parser.add_argument("--model_name", type=str, default="text-ada-001")
        parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="models/model20221028-144859_6.pt",
        )        
        parser.add_argument("--sample_size", type=int, default=8)
        parser.add_argument("--pge_avg_samples", type=int, default=10)

        parser.add_argument("--task", type=str, choices=["gsm8k","svamp","asdiv","aqua", "singleop","csqa","strategyqa","letter","obqa","esnli", "sst2"],
                            help="Indicate Task Type",default="gsm8k")
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.knowledge_dataset = KnowledgeDataset(self.hparams.knowledge_data_path)
        self.knowledge_dataset_train = KnowledgeDataset(self.hparams.train_data_path)

        self.knowledge_dataset_val = KnowledgeDataset(self.hparams.valid_data_path)

        train_size = len(self.knowledge_dataset_train)
        # Stores probabilities for each example

        self.sample_probs = torch.FloatTensor([[1 / len(self.knowledge_dataset) * 8] * int(len(self.knowledge_dataset)/8)] * self.hparams.sample_size) 
        #self.sample_probs = torch.load("./models/xxx.pt")
        self.sample_probs.requires_grad = True
        
        # Activates manual optimization
        self.automatic_optimization = False
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = True
        self.sample_probs.requires_grad = True

        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.count = 0
        print('The following parameters are trained:')
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n)
                
        self.model_filename = self.hparams.dirpath + '/model' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + "_" + str(self.count) + '.pt'
        

    def train_dataloader(self, shuffle=False):
        return DataLoader(
            self.knowledge_dataset_train,
            batch_size=self.hparams.batch_size,
            collate_fn=self.knowledge_dataset_train.collate_fn,
            shuffle=shuffle,
            num_workers = 10,
            drop_last=True
        )
        
    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.knowledge_dataset_val,
            batch_size=self.hparams.batch_size,
            collate_fn=self.knowledge_dataset_val.collate_fn,
            shuffle=shuffle,
            num_workers = 10,
            drop_last=True
        )

    def complete_gpt3(self, *args, **kwargs):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False

        while not received:
            try:
                response = openai.Completion.create(*args, **kwargs)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:  # something is wrong: e.g. prompt too long
                    print(
                        f"InvalidRequestError\nPrompt passed in:\n\n{kwargs['prompt']}\n\n")
                    assert False

                print("API error:", error)
                time.sleep(5)
        return response

    # assuming only one target for one prompt, added to the end of the prompt
    def forward(self, prompts, targets):
        rsp = self.complete_gpt3(
            model="text-davinci-002",
            prompt=prompts,
            max_tokens=256,
            temperature = 0,
            logprobs = 5
        )

        labels = []
        preds, all_losses = [], []
        match = 0
        for i , target in enumerate(targets):
            text = rsp['choices'][i]["text"]

            if(self.hparams.task == "gsm8k"):
                pattern = "The answer is \d{1,}\."
                sttr = re.search(pattern, text.replace("$","").replace(",","").replace("%",""))
                if (sttr is not None):
                    #check if match the ground truth
                    if(sttr.group(0)[14:-1] == target.replace(",","")):
                        match += 1
                        token = tokenizer(" " + sttr.group(0)[14:-1])
                        if((" " + sttr.group(0)[14:-1]) in rsp['choices'][i]["logprobs"]["tokens"]):
                            ans_index = rsp['choices'][i]["logprobs"]["tokens"].index(" "+ sttr.group(0)[14:-1])
                            prob = np.exp(rsp['choices'][i]["logprobs"]["token_logprobs"][ans_index])
                            loss = -np.log(prob) * FACTOR
                            all_losses.append(loss)
                        elif(tokenizer.decode(token['input_ids'][0]) in rsp['choices'][i]["logprobs"]["tokens"]):
                            ans_index = rsp['choices'][i]["logprobs"]["tokens"].index(tokenizer.decode(token['input_ids'][0]))
                            prob = np.exp(rsp['choices'][i]["logprobs"]["token_logprobs"][ans_index])
                            loss = -np.log(prob) * FACTOR
                            all_losses.append(loss)
                        else:
                            loss = -np.log(0.9) * FACTOR
                            all_losses.append(loss)
                    else:
                        loss = -np.log(0.05) * FACTOR
                        all_losses.append(loss)                           
                else:
                    loss = -np.log(0.05) * FACTOR
                    all_losses.append(loss)                    
            elif(self.hparams.task == "csqa"):
                pattern = "So the answer is \([a-z|A-Z]\)."
                sttr = re.search(pattern, text)
                if (sttr is not None):
                    #check if match the ground truth
                    if(sttr.group(0)[-3:-2].lower() == target.lower()):
                        match += 1
                        if( sttr.group(0)[-3:-2].lower() in rsp['choices'][i]["logprobs"]["tokens"]):
                            ans_index = rsp['choices'][i]["logprobs"]["tokens"].index(sttr.group(0)[-3:-2].lower())
                            prob = np.exp(rsp['choices'][i]["logprobs"]["token_logprobs"][ans_index])
                            loss = -np.log(prob) * FACTOR
                            all_losses.append(loss)
                        else:
                            loss = -np.log(0.9) * FACTOR
                            all_losses.append(loss)
                    else:
                        loss = -np.log(0.05) * FACTOR
                        all_losses.append(loss)                
                else:
                    loss = -np.log(0.05) * FACTOR
                    all_losses.append(loss)           
            elif(self.hparams.task == "strategyqa"):
                pattern = "So the answer is (yes|no)."
                sttr = re.search(pattern, text)
                if (sttr is not None):
                    if(sttr.group(0)[17:-1].lower() == target.lower()):
                        match += 1
                        if( " " + sttr.group(0)[17:-1].lower() in rsp['choices'][i]["logprobs"]["tokens"]):
                            ans_index = rsp['choices'][i]["logprobs"]["tokens"].index(" " + sttr.group(0)[17:-1].lower())
                            prob = np.exp(rsp['choices'][i]["logprobs"]["token_logprobs"][ans_index])
                            loss = -np.log(prob) * FACTOR
                            all_losses.append(loss)
                        else:
                            loss = -np.log(0.9) * FACTOR
                            all_losses.append(loss)
                    else:
                        loss = -np.log(0.05) * FACTOR
                        all_losses.append(loss)    
                else:
                    loss = -np.log(0.05) * FACTOR
                    all_losses.append(loss) 
            elif(self.hparams.task == "letter"):
                pattern = "So the answer is [a-zA-Z]+."
                sttr = re.search(pattern, text)
                if (sttr is not None):
                    if(sttr.group(0)[17:-1].lower() == target.lower()):
                        match += 1
                        if( " " + sttr.group(0)[17:-1].lower() in rsp['choices'][i]["logprobs"]["tokens"]):
                            ans_index = rsp['choices'][i]["logprobs"]["tokens"].index(" " + sttr.group(0)[17:-1].lower())
                            prob = np.exp(rsp['choices'][i]["logprobs"]["token_logprobs"][ans_index])
                            loss = -np.log(prob) * FACTOR
                            all_losses.append(loss)
                        elif(" " + sttr.group(0)[17:-2].lower() in rsp['choices'][i]["logprobs"]["tokens"]):
                            ans_index = rsp['choices'][i]["logprobs"]["tokens"].index(" " + sttr.group(0)[17:-2].lower())
                            prob = np.exp(rsp['choices'][i]["logprobs"]["token_logprobs"][ans_index])
                            loss = -np.log(prob) * FACTOR
                            all_losses.append(loss)                    
                        else:
                            loss = -np.log(0.9) * FACTOR
                            all_losses.append(loss) 
                    else:
                        loss = -np.log(0.05) * FACTOR
                        all_losses.append(loss)   
                else:
                    loss = -np.log(0.05) * FACTOR
                    all_losses.append(loss)  
        return np.mean(all_losses), match/10
        
    def training_step(self, batch, batch_idx=None):
        
        knowledge_data = batch
        opt = self.optimizers()
        opt.zero_grad()
        with torch.no_grad():
            # For each k, we prepend a prompt and calculate loss
            prompts_dist = torch.distributions.Categorical(self.sample_probs)
            #print(prompts_dist)
            prompts_indices_list = []
            loss_list = []
            #prompts, targets = knowledge_data['prompt'], knowledge_data['targets']
            question, rationale, answer, ground_truth = knowledge_data['Question'],knowledge_data['Rationale'],knowledge_data['Answer'],knowledge_data['Ground_truth']

            prompts = []
            targets = []

            for i in range(0,len(question)):
                if(self.hparams.task == "strategyqa"):
                    prompts.append("Q: Yes or no: " + question[i] + "\n" + "A:")
                elif(self.hparams.task == "letter"):
                    prompts.append(question[i])
                else:
                    prompts.append("Q: " + question[i] + "\n" + "A:")
                targets.append(ground_truth[i])

            for _ in range(self.hparams.pge_avg_samples):
                new_prompts = prompts
                prompt_idx = prompts_dist.sample()
                prompts_indices_list.append(copy.deepcopy(prompt_idx))


                for k in range(0,self.hparams.sample_size):
                    if(self.hparams.task == "gsm8k"):
                        prompt = "Question: " + self.knowledge_dataset[prompt_idx[k]]['Question'] + "\nAnswer:" + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " The answer is " + self.knowledge_dataset[prompt_idx[k]]['Ground_truth'] + ".\n\n"
                    elif(self.hparams.task == "csqa"):
                        prompt = "Q: " + self.knowledge_dataset[prompt_idx[k]]['Question'] + "\nA:" + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " So the answer is (" + self.knowledge_dataset[prompt_idx[k]]['Ground_truth'].lower() + ")" + ".\n\n"
                    elif(self.hparams.task == "strategyqa"):
                        if(k == 4 or k == 5):
                            prompt = "Q: " + self.knowledge_dataset[prompt_idx[k]]['Question'] + "\nA:" + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " " + self.knowledge_dataset[prompt_idx[k]]['Answer']+ "\n\n"
                        else:
                            prompt = "Q: Yes or no: " + self.knowledge_dataset[prompt_idx[k]]['Question'] + "\nA:" + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " " + self.knowledge_dataset[prompt_idx[k]]['Answer']+ "\n\n"
                    elif(self.hparams.task == "letter"):
                        prompt = self.knowledge_dataset[prompt_idx[k]]['Question']  + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " " + self.knowledge_dataset[prompt_idx[k]]['Answer'] + "\n\n"

                    new_prompts = [(prompt + x).strip() for x in new_prompts]

                # Compute loss
                losses, acc = self(new_prompts, targets)
                self.count = self.count + 1

                loss = np.mean(losses)
                loss_list.append(loss)
            

            loss_avg = sum(loss_list) / len(loss_list)
            self.log('loss_avg', loss_avg, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch['id']))

            derivative = [-1 / self.sample_probs] * self.hparams.pge_avg_samples
            # print("Begin print derivative")
            # print(derivative)
            print(prompts_indices_list)
            for k, indice in enumerate(prompts_indices_list):
                for i in range(0,self.hparams.sample_size):
                    #derivative[k // self.hparams.sample_size][indice] *= -1
                    derivative[k][i][indice[i]] *= -1

            self.sample_probs.grad = torch.zeros_like(self.sample_probs)
            for k in range(self.hparams.pge_avg_samples):
                self.sample_probs.grad += 1 / (self.hparams.pge_avg_samples - 1) * (loss_list[k] - loss_avg) * derivative[k]

            torch.nn.utils.clip_grad_norm_(self.sample_probs, 3)

            opt.step()

            # print(self.sample_probs)
            self.constrain_score_by_whole_exact(self.sample_probs)

            # print(self.sample_probs)



        # return torch.tensor(acc)
    def validation_step(self, batch, batch_idx=None):
        knowledge_data = batch

        # For each k, we prepend a prompt and calculate loss
        #prompts_dist = torch.distributions.Categorical(self.sample_probs)
        prompts_indices_list = []
        
        loss_list = []
        #prompts, targets = knowledge_data['prompt'], knowledge_data['targets']
        question, rationale, answer, ground_truth = knowledge_data['Question'],knowledge_data['Rationale'],knowledge_data['Answer'],knowledge_data['Ground_truth']
        prompts = []
        targets = []
        for i in range(0,len(question)):
            if(self.hparams.task == "strategyqa"):
                prompts.append("Q: Yes or no: " + question[i] + "\n" + "A:")
            elif(self.hparams.task == "letter"):
                prompts.append(question[i])
            else:
                prompts.append("Q: " + question[i] + "\n" + "A:")            
            targets.append(ground_truth[i])
        # Create a tensor of size [prompt_length]
        prompt_idx = torch.zeros(len(self.sample_probs),dtype=torch.int64) 
        # Fill it with argmax for each dimension
        for i in range(0,len(self.sample_probs)):
            # Retrieve argmax index
            idx = (self.sample_probs[i]==max(self.sample_probs[i])).nonzero().squeeze()
            # If only one index are retrieved 
            if(idx.size() == torch.Size([])):
                prompt_idx[i] = idx
            else:
                prompt_idx[i] = idx[0]

        prompts_indices_list.append(copy.deepcopy(prompt_idx))
        for k in range(0,self.hparams.sample_size):
            # Sample a prompt
   
            if(self.hparams.task == "gsm8k"):
                prompt = "Question: " + self.knowledge_dataset[prompt_idx[k]]['Question'] + "\nAnswer: Let's think step by step." + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " The answer is " + self.knowledge_dataset[prompt_idx[k]]['Ground_truth'] + ".\n\n"
            elif(self.hparams.task == "csqa"):
                prompt = "Q: " + self.knowledge_dataset[prompt_idx[k]]['Question'] + "\nA:" + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " So the answer is (" + self.knowledge_dataset[prompt_idx[k]]['Ground_truth'].lower() + ")" + ".\n\n"
            elif(self.hparams.task == "strategyqa"):
                if(k == 4 or k == 5):
                    prompt = "Q: " + self.knowledge_dataset[prompt_idx[k]]['Question'] + "\nA:" + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " " + self.knowledge_dataset[prompt_idx[k]]['Answer']+ "\n\n"
                else:
                    prompt = "Q: Yes or no: " + self.knowledge_dataset[prompt_idx[k]]['Question'] + "\nA:" + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " " + self.knowledge_dataset[prompt_idx[k]]['Answer']+ "\n\n"
            elif(self.hparams.task == "letter"):
                prompt = self.knowledge_dataset[prompt_idx[k]]['Question']  + self.knowledge_dataset[prompt_idx[k]]['Rationale'] + " " + self.knowledge_dataset[prompt_idx[k]]['Answer'] + "\n\n"
         
            prompts = [(prompt + x).strip() for x in prompts]

        # Compute loss
        losses,acc = self(prompts, targets)
        loss = np.mean(losses)

        loss_avg = loss
        print("One Validation Steo End. Evaluating Result : \n")
        print("Validation Acc{}".format(acc))

        # self.log('valid_acc', valid_acc, prog_bar=True, on_step=True,
        #          on_epoch=True, batch_size=len(batch['id']))
        return torch.tensor(acc)

    def configure_optimizers(self):
        optimizer = Adam([self.sample_probs], lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=0)
        return [optimizer], [scheduler]


    # def training_epoch_end(self, outputs) -> None:
    #     epoch_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     print("\nEpoch {} Train acc is {}\n".format(self.current_epoch, epoch_loss))
    
    def validation_epoch_end(self, outputs) -> None:
        valid_acc = torch.stack(outputs).mean()
        print("\n Epoch {} Valid acc is {} \n".format(self.current_epoch, valid_acc))
        print("\n")
    
    def on_train_epoch_end(self):
        print(self.sample_probs)
        print("Trian epoch end, saving Model")
        self.model_filename = self.hparams.dirpath + '/model' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + "_" + str(self.count) + '.pt'
        #torch.save(self.sample_probs, self.model_filename)
    
    def solve_v_total_exact(self, prompt_emb):
        k = 1
        a, b = 0, 0

        b = prompt_emb.max()
        def f(v):
            s = (prompt_emb - v).clamp(0.0001, 1).sum()
            return s - k
        itr = 0

        v = 0
        while True:
            itr += 1
            v = (a + b) / 2
            obj = f(v)
            if abs(obj) < 1e-3 or itr > 30:
                break
            if obj < 0:
                b = v
            else:
                a = v
        return v, itr

    def constrain_score_by_whole_exact(self, prompt_embeds):
        for i in range(len(prompt_embeds)):
            v, itr = self.solve_v_total_exact(prompt_embeds[i])
            prompt_embeds[i].sub_(v).clamp_(0.0001, 1)

    # See bilevel training paper p. 14
    def project(self, z: torch.FloatTensor):
        v = max(0, (z.sum() - 1) / z.numel())
        z = z.add(-v)
        z = z.clamp(0, 1)
        assert 0 <= z.sum() <= 1
        return z
    