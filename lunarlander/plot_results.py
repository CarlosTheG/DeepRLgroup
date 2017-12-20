import matplotlib.pyplot as plt

file1 = 'DeepQALearn_results.txt'
file2 = 'DeepQLearn_results.txt'

def parse_arr_line(line):
    words = line.split(',')
    length = len(words)
    out = []
    for i in range(len(words)):
        word = words[i]
        if i == length - 1:
            out.append(float(word[:-2]))
        elif i == 0:
            out.append(float(word[1:]))
        else:
            out.append(float(word))
    return out

with open(file1) as f:
    content = f.readlines()

rewards1 = parse_arr_line(content[0])
loss1 = parse_arr_line(content[1])

with open(file2) as f:
    content2 = f.readlines()

rewards2 = parse_arr_line(content2[0])
loss2 = parse_arr_line(content2[1])[:len(loss1)]

# plt.plot(range(len(rewards1)), rewards1, label="DeepQA")
# plt.plot(range(len(rewards2)), rewards2, label="DeepQ")
# plt.legend()
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.savefig("qa_vs_q_reward.png")

plt.plot(range(len(loss1)), loss1, label="DeepQA")
plt.plot(range(len(loss2)), loss2, label="DeepQ")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("qa_vs_q_loss.png")
