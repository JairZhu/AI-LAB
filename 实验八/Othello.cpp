#include <iostream>
#include <time.h>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <map>
using namespace std;

struct Node {
	int mode;
	char board[8][8];
	double score;
	pair<int, int> action;
	vector<Node> children;
	Node(char raw_board[8][8], int cur_mode) {
		mode = cur_mode;
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
				board[i][j] = raw_board[i][j];
	}
};

class Othello {
private:
	int square_weights[8][8] = {
		{20, -3, 11, 8, 8, 11, -3, 20},
		{-3, -7, -4, 1, 1, -4, -7, -3},
		{11, -4, 2, 2, 2, 2, -4, 11},
		{8, 1, 2, -3, -3, 2, 1, 8},
		{8, 1, 2, -3, -3, 2, 1, 8},
		{11, -4, 2, 2, 2, 2, -4, 11},
		{-3, -7, -4, 1, 1, -4, -7, -3},
		{20, -3, 11, 8, 8, 11, -3, 20},
	};
	int direction[8][2] = { {0, 1}, {0, -1}, {1, 1}, {1, 0}, {1, -1}, {-1, 0}, {-1, 1}, {-1, -1} };
	int limit;

public:
	Othello(int depth) { limit = depth; }

	void init_board(char board[8][8]) {
		for (int i = 0; i < 8; ++i)
			memset(board[i], ' ', 8);
		board[3][4] = board[4][3] = '@';
		board[3][3] = board[4][4] = 'O';
	}

	void print_board(char board[8][8], vector<pair<int, int>> next_step) {
		char col = 'A';
		cout << endl << ' ';
		for (char i = 0; i < 8; i++)
			cout << ' ' << char(col + i);
		cout << endl;
		for (int i = 0; i < 8; i++) {
			cout << i + 1 << ' ';
			for (int j = 0; j < 8; ++j)
				if (find(next_step.begin(), next_step.end(), make_pair(i, j)) == next_step.end())
					cout << board[i][j] << ' ';
				else
					cout << "* ";
			cout << endl;
		}
		auto nums = count_number(board);
		cout << "����:���� = " << nums.first << ':' << nums.second << endl;
	}

	bool is_on_board(int row, int col) {
		return row < 8 && row >= 0 && col < 8 && col >= 0;
	}

	vector<pair<int, int>> show_places(char board[8][8], char turn) {
		vector<pair<int, int>> next_step;
		char item = turn == '@' ? '@' : 'O', opp = turn == '@' ? 'O' : '@';
		for (int i = 0; i < 8; ++i)
			for (int j = 0; j < 8; ++j)
				if (board[i][j] == ' ' && !flip_dirs(board, make_pair(i, j), item).empty())
					next_step.push_back(make_pair(i, j));
		return next_step;
	}

	vector<int> flip_dirs(char board[8][8], pair<int, int> pos, char color) {
		int x = pos.first, y = pos.second;
		vector<int> dirs;
		char opp = color == '@' ? 'O' : '@';
		for (int i = 0; i < 8; ++i) {
			x += direction[i][0];
			y += direction[i][1];
			if (is_on_board(x, y) && board[x][y] == opp) {
				while (is_on_board(x, y) && board[x][y] == opp) {
					x += direction[i][0];
					y += direction[i][1];
				}
				if (is_on_board(x, y) && board[x][y] == color) dirs.push_back(i);
			}
			x = pos.first;
			y = pos.second;
		}
		return dirs;
	}

	void flip(char board[8][8], pair<int, int> action, char color) {
		vector<int> dirs = flip_dirs(board, action, color);
		char opp = color == '@' ? 'O' : '@';
		int x = action.first, y = action.second;
		for (int i = 0; i < dirs.size(); ++i) {
			x += direction[dirs[i]][0];
			y += direction[dirs[i]][1];
			while (is_on_board(x, y) && board[x][y] == opp) {
				board[x][y] = color;
				x += direction[dirs[i]][0];
				y += direction[dirs[i]][1];
			}
			x = action.first;
			y = action.second;
		}
	}

	bool is_over(char board[8][8]) {
		auto list1 = show_places(board, '@');
		auto list2 = show_places(board, 'O');
		return list1.empty() && list2.empty();
	}

	void player_play(char board[8][8], vector<pair<int, int>> next_step, char turn) {
		cout << "���������꣨���磺A1����";
		while (true) {
			char col, row;
			cin >> col >> row;
			if (col >= 'A' && col <= 'H' && row >= '1' && row <= '8') {
				auto action = make_pair(row - '1', col - 'A');
				if (find(next_step.begin(), next_step.end(), action) != next_step.end()) {
					move(board, action, turn);
					break;
				}
			}
			cout << "��Чλ��,���������루���磺A1����";
		}
		cout << endl;
	}

	double evaluate(char board[8][8], char color) {
		int sideVal[9] = { 1, 1, 1, 2, 3, 4, 6, 7 };
		int mystonecount = 0, opstonecount = 0;
		double score = 0, rateeval = 0, moveeval = 0, sidestableeval = 0, cornereval = 0;
		char opp = color == '@' ? 'O' : '@';
		//����λ����������ֵ
		for (int i = 0; i < 8; ++i)
			for (int j = 0; j < 8; ++j)
				if (board[i][j] == color) {
					score += square_weights[i][j];
					mystonecount++;
				}
				else if (board[i][j] == opp) {
					score -= square_weights[i][j];
					opstonecount++;
				}
		//����ڰ��ӱ�������ֵ
		if (mystonecount > opstonecount)
			rateeval = 100.0 * mystonecount / (mystonecount + opstonecount);
		else if (mystonecount < opstonecount)
			rateeval = -100.0 * opstonecount / (mystonecount + opstonecount);
		else
			rateeval = 0;
		//�����ж�������ֵ
		int mymove = show_places(board, color).size();
		int opmove = show_places(board, opp).size();
		//����ҷ�û�еط����ӣ���ô�趨�ص͵��ж�������ֵ
		if (mymove == 0)
			moveeval = -450;
		//����Է�û�еط����ӣ���ô�趨�ظߵ��ж�������ֵ
		else if (opmove == 0)
			moveeval = 150;
		else if (mymove > opmove)
			moveeval = (100.0 * mymove) / (mymove + opmove);
		else if (mymove < opmove)
			moveeval = -(100.0 * opmove) / (mymove + opmove);
		else
			moveeval = 0;
		//����߽��ȶ������ֵ
		int myside = 0, opside = 0, myconer = 0, opconer = 0;
		int corner_pos[4][2] = { {0, 0}, {0, 7}, {7, 0}, {7, 7} };
		for (int i = 0; i < 4; ++i)
			if (board[corner_pos[i][0]][corner_pos[i][1]] == color) {
				myconer++;
				for (int j = 0; j < 8; ++j)
					if (board[corner_pos[i][0]][j] == color)
						myside += sideVal[i];
					else
						break;
				for (int j = 0; j < 8; ++j)
					if (board[j][corner_pos[i][1]] == color)
						myside += sideVal[i];
					else
						break;
			}
			else if (board[corner_pos[i][0]][corner_pos[i][1]] == opp) {
				opconer++;
				for (int j = 0; j < 8; ++j)
					if (board[corner_pos[i][0]][j] == opp)
						opside += sideVal[i];
					else
						break;
				for (int j = 0; j < 8; ++j)
					if (board[j][corner_pos[i][1]] == opp)
						opside += sideVal[i];
					else
						break;
			}
		sidestableeval = 2.5 * (myside - opside);
		cornereval = 25 * (myconer - opconer);
		//����������ֹ���ֵ
		return score * 0.02 + moveeval * 1 + sidestableeval * 6.0 + cornereval * 8.0 + rateeval * 0.2;
	}

	void move(char board[8][8], pair<int, int> action, char color) {
		board[action.first][action.second] = color;
		flip(board, action, color);
	}

	pair<int, int> count_number(char board[8][8]) {
		int b_num = 0, w_num = 0;
		for (int i = 0; i < 8; ++i)
			for (int j = 0; j < 8; ++j)
				if (board[i][j] == '@')
					b_num++;
				else if (board[i][j] == 'O')
					w_num++;
		return make_pair(b_num, w_num);
	}

	void random_play(char board[8][8], char color) {
		auto places = show_places(board, color);
		int index = rand() % places.size();
		move(board, places[index], color);
	}

	void ai_play(char board[8][8], char color, int depth) {
		char opp = color == '@' ? 'O' : '@';
		Node root = Node(board, 2);
		cout << "\nIDIOT����˼��...\n";
		double starttime = clock();
		int index = alphabetapruning(root, color, opp, 1, depth, -100000.0, 100000.0);
		cout << "IDIOT������λ��Ϊ��" << char(root.children[index].action.second + 'A') << root.children[index].action.first + 1 << endl;
		double endtime = clock();
		cout << "IDIOT˼����ʱ��" << (endtime - starttime) / 1000 << 's' << endl;
		move(board, root.children[index].action, color);
		root.children.clear();
		cout << endl;
	}

	double alphabetapruning(Node &root, char ai, char player, int mode, int depth, double alpha, double beta) {
		//mode=0ʱ��ʾMAX��ڵ㣬mode=1ʱ��ʾMIN��ڵ�
		char color = mode == 1 ? ai : player, opp = color == '@' ? 'O' : '@';
		auto avaiplaces = show_places(root.board, color);	//�õ������ӵ�λ��
		double v;
		if (depth == limit) {
			for (int i = 0; i < avaiplaces.size(); ++i) {
				Node newnode = Node(root.board, mode);		//�½��ӽڵ�
				newnode.action = avaiplaces[i];				//��¼�ýڵ����ӵ�λ��
				move(newnode.board, avaiplaces[i], color);	//���Ӻ����̷����仯
				int oppmode = mode == 1 ? 0 : 1;			//������һ��mode
				auto places = show_places(newnode.board, opp);	//�õ��Է������ӵ�λ��
				if (places.size() != 0)
					//����Է����п����ӵ�λ�ã���ݹ�����
					newnode.score = alphabetapruning(newnode, ai, player, oppmode, depth - 1, alpha, beta);
				else
					//����Է�û�еط����ӣ���������ǰ���
					newnode.score = evaluate(newnode.board, color);
				root.children.push_back(newnode);			//�����µ��ӽڵ�
			}
			int index;
			double max = -100000.0;
			//�õ�����ֵ��ߵ��߷�
			for (int i = 0; i < root.children.size(); ++i)
				if (root.children[i].score > max) {
					index = i;
					max = root.children[i].score;
				}
			cout << "IDIOT���߷�����Ӧ�Ĺ���ֵ��";
			for (int i = 0; i < root.children.size(); ++i) {
				cout << char(root.children[i].action.second + 'A') << root.children[i].action.first + 1 << ':' << root.children[i].score << "  ";
			}
			cout << endl;
			//���ع���ֵ��ߵ��߷�
			return index;
		}
		if (mode == 0) {
			//MAX��ڵ�
			v = -100000.0;
			for (int i = 0; i < avaiplaces.size(); ++i) {
				Node newnode = Node(root.board, mode);		//�½��ӽڵ�
				newnode.action = avaiplaces[i];				//��¼�ýڵ����ӵ�λ��
				move(newnode.board, avaiplaces[i], color);	//���Ӻ����̷����仯
				int oppmode = mode == 1 ? 0 : 1;			//������һ��mode
				auto places = show_places(newnode.board, opp);	//�õ��Է������ӵ�λ��
				if (depth != 1 && places.size() != 0) {
					//δ������������ҶԷ��еط����ӣ���ݹ�����������v��alphaֵ
					v = max(v, alphabetapruning(newnode, ai, player, oppmode, depth - 1, alpha, beta));
					alpha = max(alpha, v);
					if (beta <= alpha)	//alpha��֦
						break;
				}
				else {
					//����������ƻ�Է����ӿ��£���������ǰ��ֲ�����vֵ
					newnode.score = evaluate(newnode.board, ai);
					v = max(v, newnode.score);
				}
			}
		}
		else {
			//MIN��ڵ�
			v = 100000.0;
			for (int i = 0; i < avaiplaces.size(); ++i) {
				Node newnode = Node(root.board, mode);		//�½��ӽڵ�
				newnode.action = avaiplaces[i];;			//��¼�ýڵ����ӵ�λ��
				move(newnode.board, avaiplaces[i], color);	//���Ӻ����̷����仯
				int oppmode = mode == 1 ? 0 : 1;			//������һ��mode
				auto places = show_places(newnode.board, opp);	//�õ��Է������ӵ�λ��
				if (depth != 1 && places.size() != 0) {
					//δ������������ҶԷ��еط����ӣ���ݹ�����������v��betaֵ
					v = min(v, alphabetapruning(newnode, ai, player, oppmode, depth - 1, alpha, beta));
					beta = min(beta, v);
					if (beta <= alpha)	//beta��֦
						break;
				}
				else {
					//����������ƻ�Է����ӿ��£���������ǰ��ֲ�����vֵ
					newnode.score = evaluate(newnode.board, ai);
					v = min(v, newnode.score);
				}
			}
		}
		//����ʱ���ռ�õ��ڴ�
		root.children.clear();
		return v;
	}

	int game_run() {
		cout << "��ѡ��1Ϊ���壬2Ϊ���壩��";
		int choose = 1;
		cin >> choose;
		char player = choose == 1 ? '@' : 'O', ai = choose == 1 ? 'O' : '@';
		cout << "\n��Ϸ��ʼ��\n@Ϊ���壬OΪ����,*Ϊ������λ��\n\n";
		char current_turn = '@';
		char board[8][8];
		vector<pair<int, int>> next_step;
		init_board(board);
		while (!is_over(board)) {
			next_step = show_places(board, current_turn);
			if (next_step.empty()) {
				current_turn = current_turn == '@' ? 'O' : '@';
				next_step.clear();
				continue;
			}
			if (current_turn == player) {
				print_board(board, next_step);
				cout << "\n�ֵ���ң�";
				player_play(board, next_step, current_turn);
				//random_play(board, player);
			}
			else {
				print_board(board, next_step);
				ai_play(board, ai, limit);
			}
			next_step.clear();
			current_turn = current_turn == '@' ? 'O' : '@';
		}
		auto nums = count_number(board);
		print_board(board, next_step);
		cout << "��Ϸ������" << endl;
		if (nums.first > nums.second) {
			cout << "����ʤ����" << endl;
			return 0;
		}
		else if (nums.first < nums.second) {
			cout << "����ʤ����" << endl;
			return 1;
		}
		else {
			cout << "ƽ�֣�" << endl;
			return 0;
		}
	}
};


int main() {
	auto tmp = Othello(4);
	tmp.game_run();
	system("pause");
	return 0;
}