import torch
from navi_config import NaviConfig
from utils import make_cfg,sim_settings
import numpy as np
import cv2
import habitat_sim
from PIL import Image
from modelscope import AutoModel, AutoTokenizer

def llm_speak(image,question):
    answer = llm.chat(
            image=image,
            msgs=[{'role': 'user', 'content': question}],
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            # system_prompt='' # pass system_prompt if needed
        )
    
    return answer

# w,a,d,q for move forward, turn left, turn right, quit

def hook_keystroke(image) -> str:
    '''
    capture keystroke and outputs corresponding action string
    '''
    while True:
        key_stroke = cv2.waitKey(0)
        if key_stroke == ord('w'):
            return 'move_forward'
        elif key_stroke == ord('a'):
            return 'turn_left'
        elif key_stroke == ord('d'):
            return 'turn_right'
        elif key_stroke == ord('q'):
            cv2.destroyAllWindows()
            return 'terminate'
        elif key_stroke == ord('l'):
            while True:
                question=input('your question?')
                if question=='q':
                    break

                answer=llm_speak(question,image)
                print('LLM: ',answer)

if __name__=='__main__':
    # change
    env_path=f'envs/mp3d_example/HaxA7YrQdEC.basis.glb'


    SPLs=[]
    sim_settings["scene"] = env_path
    cfg = make_cfg(sim_settings)

    simulator = habitat_sim.Simulator(cfg)

    # reset
    agent = simulator.initialize_agent(sim_settings["default_agent"])
    
    # Set agent state if needed
    # agent_state = habitat_sim.AgentState()
    # agent_state.position = np.array(start_pos)  # in world space
    # agent_state.rotation = np.array(start_rot)
    # agent.set_state(agent_state)
    
    action='turn_right'

    max_step=100
    step=0

    llm = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
    llm = llm.to(device=f'cuda:{NaviConfig.GPU_ID}')

    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
    llm.eval()


    while True:
        obs=simulator.step(action)
        rgb_image = obs['color_sensor'][..., :3][..., ::-1]
        cv2.imshow('observation',np.uint8(rgb_image))
        
        image=Image.fromarray(np.uint8(rgb_image),'RGB')
        action=hook_keystroke(image)
        
        # record agent state
        agent_state = agent.get_state()
        now_pos=agent_state.position
        print('position:', now_pos)
        step+=1

        if action=='terminate' or step==max_step:
            break

    
    simulator.close()
    
    exit()