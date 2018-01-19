# Libraries
-  Tensorflow Agents
    - [https://github.com/tensorflow/agents/](https://github.com/tensorflow/agents/)
- Simulation Environment
    - [OpenAI Gym](https://github.com/openai/gym)   
        For those unfamiliar, the OpenAI gym provides an easy way for people to experiment 
with their learning agents in an array of provided toy games.
    - [MuJoCo](https://www.roboti.us/)     
      [Mujoco-py](https://github.com/openai/mujoco-py)   
        Setup Notes:  
            This requires an college email-id to get an free version!  
            `mjpro131 works fine`
            
            ```
            sudo apt-get install libffi-dev
            sudo apt-get install libglew-dev
            sudo apt install patchelf
            
            cd /path/to/mjpro150/bin
            mkdir ~/.mujoco/mjpro150
            cp * ~/.mujoco/mjpro150
            export LD_LIBRARY_PATH=/path/to/mjpro150/bin/:$LD_LIBRARY_PATH
            ./simulate ../model/humanoid.xml
            ```
### References:
- https://github.com/ashutoshkrjha/Cartpole-OpenAI-Tensorflow