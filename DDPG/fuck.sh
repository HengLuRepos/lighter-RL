ffmpeg -i video/DDPG-HalfCheetah-v4-baseline-episode-0.mp4 -i video/DDPG-HumanoidStandup-v4-baseline-episode-0.mp4 -i video/DDPG-Ant-v4-baseline-episode-0.mp4 -i video/DDPG-Humanoid-v4-baseline-episode-0.mp4 -i video/DDPG-Hopper-v4-baseline-episode-0.mp4 -i video/DDPG-HalfCheetah-v4-dynamic-episode-0.mp4 -i video/DDPG-HumanoidStandup-v4-qat-episode-0.mp4 -i video/DDPG-Ant-v4-static-episode-0.mp4 -i video/DDPG-Humanoid-v4-qat-episode-0.mp4 -i video/DDPG-Hopper-v4-dynamic-episode-0.mp4 -i video/DDPG-HalfCheetah-v4-pruning-episode-0.mp4 -i video/DDPG-HumanoidStandup-v4-pruning-episode-0.mp4 -i video/DDPG-Ant-v4-pruning-episode-0.mp4 -i video/DDPG-Humanoid-v4-pruning-episode-0.mp4 -i video/DDPG-Hopper-v4-pruning-episode-0.mp4 -filter_complex "color=c=black:size=2880x1920[bg]; \
[bg][0:v]overlay=W/6:H/4[v0]; \
[v0][1:v]overlay=2*W/6:H/4[v1]; \
[v1][2:v]overlay=3*W/6:H/4[v2]; \
[v2][3:v]overlay=4*W/6:H/4[v3]; \
[v3][4:v]overlay=5*W/6:H/4[v4]; \
[v4][5:v]overlay=W/6:2*H/4[v5]; \
[v5][6:v]overlay=2*W/6:2*H/4[v6]; \
[v6][7:v]overlay=3*W/6:2*H/4[v7]; \
[v7][8:v]overlay=4*W/6:2*H/4[v8]; \
[v8][9:v]overlay=5*W/6:2*H/4[v9]; \
[v9][10:v]overlay=W/6:3*H/4[v10]; \
[v10][11:v]overlay=2*W/6:3*H/4[v11]; \
[v11][12:v]overlay=3*W/6:3*H/4[v12]; \
[v12][13:v]overlay=4*W/6:3*H/4[v13]; \
[v13][14:v]overlay=5*W/6:3*H/4, \
drawtext=text='DDPG':x=(W/12)-(text_w/2):y=(H/8)-(text_h/2):fontsize=24:fontcolor=white, \
drawtext=text='HalfCheetah':x=(3*W/12)-(text_w/2):y=(H/8)-(text_h/2):fontsize=24:fontcolor=white, \
drawtext=text='HumanoidStandup':x=(5*W/12)-(text_w/2):y=(H/8)-(text_h/2):fontsize=24:fontcolor=white, \
drawtext=text='Ant':x=(7*W/12)-(text_w/2):y=(H/8)-(text_h/2):fontsize=24:fontcolor=white, \
drawtext=text='Humanoid':x=(9*W/12)-(text_w/2):y=(H/8)-(text_h/2):fontsize=24:fontcolor=white, \
drawtext=text='Hopper':x=(11*W/12)-(text_w/2):y=(H/8)-(text_h/2):fontsize=24:fontcolor=white, \
drawtext=text='baseline':x=(W/12)-(text_w/2):y=(3*H/8)-(text_h/2):fontsize=24:fontcolor=white, \
drawtext=text='quantization':x=(W/12)-(text_w/2):y=(5*H/8)-(text_h/2):fontsize=24:fontcolor=white, \
drawtext=text='pruning':x=(W/12)-(text_w/2):y=(7*H/8)-(text_h/2):fontsize=24:fontcolor=white[v14]" -t 52 -map "[v14]" ddpg.mp4
