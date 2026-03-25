import torch
import os
import safetensors.torch
import bitsandbytes
import visualization
import tqdm
import torchvision
import tensordict
import random
import math
import plotly.graph_objects

class Framework:

    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        history: str
    ) -> None:
        self.model = model
        self.device = device
        self.history = history
        return

    def saveCheckpoint(self, weight: dict, path: str) -> bool:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # safetensors.torch.save_file(self.model.state_dict(), path)
        torch.save(weight, path)
        return(True)

    def getMemory(self) -> float:
        freeness, total = torch.cuda.mem_get_info()
        occupancy = total - freeness
        value = int(round((occupancy / total), 2) * 100)
        memory = value
        return(memory)

    def fitWeight(
        self, 
        data: torch.utils.data.DataLoader,
        snapshot: int,
        total: int,
        accumulation: int,
        validation: torch.utils.data.DataLoader,
        # schema: dict
    ) -> bool:
        # optimization
        optimization = bitsandbytes.optim.AdamW(
            self.model.parameters(), 1e-4
        )
        # schedule
        # schedule = Schedule(optimization, **schema)
        # schedule.activateStep()
        #
        # self.model = torch.compile(self.model)
        version = torch.optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
        )
        # dashboard
        dashboard = visualization.Dashboard(self.history)
        dashboard.openSession()
        #
        gradient = torch.amp.GradScaler()
        #
        number = 1
        #
        termination = False
        self.model.train()
        while(not termination):
            iteration = tqdm.tqdm(data)
            for batch in iteration:
                with torch.amp.autocast(self.device):
                    criteria = self.model(batch)
                    pass
                loss = torch.div(criteria['total'], accumulation)
                gradient.scale(loss).backward()
                if(number%accumulation==0):
                    gradient.unscale_(optimization)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=1.0
                    )
                    gradient.step(optimization)
                    gradient.update()
                    # schedule.updateStep()
                    optimization.zero_grad()
                    version.update_parameters(self.model)
                    pass
                element = {
                    'Tolerance':  criteria['tolerance'],
                    # 'Perceptual': criteria['perceptual'],
                    # 'Stillness':  criteria['stillness'],
                    'Total':      criteria['total'],
                }
                dashboard.insertStatistic('Loss/Data', element, number)
                # Validation
                self.model.eval()
                with torch.no_grad():
                    batch = next(iter(validation))
                    criteria = self.model(batch)
                    pass
                self.model.train()
                #
                element = {
                    'Tolerance':  criteria['tolerance'],
                    # 'Perceptual': criteria['perceptual'],
                    # 'Stillness':  criteria['stillness'],
                    'Total':      criteria['total'],
                }
                dashboard.insertStatistic(
                    'Loss/Validation',
                    element,
                    number
                )
                # Snapshot
                if((number==1) or (number)%snapshot==0):
                    path = os.path.join(
                        self.history, 'checkpoint', f'{number}.pt'
                    )
                    self.saveCheckpoint(self.model.state_dict(), path)
                    # version
                    path = os.path.join(
                        self.history, 'version', f'{number}.pt'
                    )
                    self.saveCheckpoint(version.state_dict(), path)
                    pass
                number += 1
                if(total==-1 or total<number): continue
                termination = True
                break
            _ = iteration
            continue
        dashboard.closeSession()
        return(True)

    def saveWeight(self, checkpoint: list) -> bool:
        # folder = os.path.join(self.history, 'weight')
        index = checkpoint.pop(0)
        path = os.path.join(self.history, 'checkpoint', index)
        aggregate = {}
        iteration = safetensors.torch.load_file(path).items()
        for key, value in iteration:
            aggregate.update({key: value.clone()})
            continue
        _ = iteration
        #
        iteration = checkpoint
        for index in iteration:
            path = os.path.join(self.history, 'checkpoint', index)
            state = safetensors.torch.load_file(path)
            for key in aggregate: aggregate[key] += state[key]
            continue
        _ = iteration
        size = len(checkpoint) + 1
        for key in aggregate: aggregate[key] /= size
        structure = self.model.state_dict()
        assert aggregate.keys() == structure.keys()
        weight = aggregate
        path = os.path.join(self.history, 'weight.st')
        safetensors.torch.save_file(weight, path)
        return(True)
    
class Schedule:

    def __init__(
        self, 
        optimization: torch.optim.Optimizer, 
        cycle: int,  # 每個週期迭代次數，也代表第一個週期的迭代次數。
        ratio: float, # 週期縮放因子。
        boundary: tuple,  # 學習率邊界，分別代表最小值與最大值。
        decay: float, # 最大學習率衰減因子。
        acceleration: int    # 從最小學習率到最大學習率會經過幾次迭代，首次啟動使用。
    ) -> None:
        self.optimization = optimization
        self.cycle = cycle
        self.ratio = ratio
        self.boundary = boundary
        self.decay = decay
        self.acceleration = acceleration
        return
    
    def activateStep(self) -> bool:
        # # 在訓練 loop 之前要執行
        # assert self.ratio > 1
        valley, peak = self.boundary
        if(self.acceleration == 0):
            rate = peak
            iteration = self.optimization.param_groups
            for group in iteration: group['lr'] = rate
            _ = iteration
            #
            acceleration = self.acceleration
            total = 0
            count = 1
            index = 0
            step = {
                'total': total,
                'acceleration': acceleration,
                'rate': rate,
                'cycle': {
                    'index': index,
                    'length': self.cycle, # 這個週期有幾個迭代要跑
                    'count': count # 目前跑了幾個迭代
                }
            }
            self.step = step
            return(True)
        rate = valley
        iteration = self.optimization.param_groups
        for group in iteration: group['lr'] = rate
        _ = iteration
        #
        acceleration = self.acceleration
        total = 0
        count = 0
        index = 0
        step = {
            'total': total,
            'acceleration': acceleration,
            'rate': rate,
            'cycle': {
                'index': index,
                'length': self.cycle, # 這個週期有幾個迭代要跑
                'count': count # 目前跑了幾個迭代
            }
        }
        self.step = step
        return(True)

    def updateStep(self) -> bool:
        total = self.step['total']
        total += 1
        if(total < self.acceleration):
            valley, peak = self.boundary
            delta = (peak - valley) * (total / (self.acceleration - 1))
            rate = valley + delta 
            pass
        else:
            length = self.step['cycle']['length']
            count = self.step['cycle']['count']
            index = self.step['cycle']['index']
            if(count >= length): # 如果都做完計算下一個 cycle 的 length
                index += 1
                length = int(length * self.ratio)
                count = 0
                pass
            valley, peak = self.boundary
            maximum = peak * (self.decay ** index)
            if(maximum <= valley): maximum = valley
            delta = 1 + math.cos(math.pi * count / (length - 1))
            rate = valley + 0.5 * (maximum - valley) * delta
            count += 1
            self.step['cycle']['count'] = count
            self.step['cycle']['length'] = length
            self.step['cycle']['index'] = index
            pass
        iteration = self.optimization.param_groups
        for group in iteration: group['lr'] = rate
        _ = iteration
        # total += 1
        self.step['rate'] = rate
        self.step['total'] = total
        return(True)

    def saveFigure(self, number: int, path: str) -> bool:
        total = []
        rate = []
        self.activateStep()
        total.append(self.step['total'])
        rate.append(self.step['rate'])
        for _ in range(number):
            self.updateStep()
            total.append(self.step['total'])
            rate.append(self.step['rate'])
            continue
        # total = list(map(lambda item: item['total'], data))
        # rate = list(map(lambda item: item['rate'], data))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        figure = plotly.graph_objects.Figure()
        figure.add_trace(plotly.graph_objects.Scatter(
            x=total, y=rate, mode='lines')
        )
        figure.update_layout(xaxis_title='iteration', yaxis_title='rate')
        figure.write_html(path)
        return(True)

    pass