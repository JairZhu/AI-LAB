#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
using namespace std;


bool constraint_check(int board[30][30], int x, int y, int n) {
	//Լ���Լ�⣬�ж����к�б�����Ƿ���ڻʺ�
	int directions[3][2] = { {-1, -1}, {-1, 0}, {-1, 1} };
	for (int i = 0; i < 3; ++i)
		for (int nx = x, ny = y; nx < n && ny < n && nx >= 0 && ny >= 0; nx += directions[i][0], ny += directions[i][1])
			if (board[nx][ny])
				return false;
	return true;
}

bool backtracking(int board[30][30], int level, int n) {
	if (level == n) 
		//�ҵ�һ���⣬��������
		return true;
	for (int i = 0; i < n; ++i)
		if (constraint_check(board, level, i, n)) {
			//����Լ�����������ʺ���õ���λ��
			board[level][i] = 1;
			if (backtracking(board, level + 1, n))
				//�������������һ��λ��
				return true;
			else 
				//�²㷵��false�����������ڽ⣬�򽫸�λ�õĻʺ��Ƴ�
				board[level][i] = 0;
		}
	//δ�ҵ�������ûʺ��λ�ã�����false
	return false;
}

bool FCCheck(int domain[30][30], int level, int y, int n, bool recovery) {
	int directions[3][2] = { {1, -1}, {1, 0}, {1, 1} };
	for (int j = 0; j < 3; ++j)
		for (int nx = level + directions[j][0], ny = y + directions[j][1]; nx < n && ny < n && ny >= 0; nx += directions[j][0], ny += directions[j][1]) {
			if (!recovery)
				//��domain�������Ӧλ��+1����ʾ��ȡֵ��Χ��ɾ����ֵ
				domain[nx][ny]++;
			else
				//��domain�������Ӧλ��-1����ֵ��ȡֵ��Χ�лָ���ֵ
				domain[nx][ny]--;
		}
	if (recovery)
		//����ǻָ����̣��������Ƿ�DWO
		return true;
	for (int i = 0; i < n; ++i) {
		bool empty = true;
		for (int j = 0; j < n; ++j) 
			if (domain[i][j] == 0) {
				empty = false;
				break;
			}
		if (empty) 
			//����ĳ��������ȡֵ��ΧΪ�ռ�������DWO
			return false;
	}
	//������ȡֵ��ΧΪ�ռ��ı���
	return true;
}

bool FC(int board[30][30], int domain[30][30], int level, int n) {
	if (level == n)
		//�ҵ�һ���⣬��������
		return true;
	for (int i = 0; i < n; ++i) {
		if (domain[level][i] == 0) {
			//���domain��������Ӧλ�õ��ϵ�ֵΪ0������Է��ûʺ�
			board[level][i] = 1;
			if (FCCheck(domain, level, i, n, false))
				//ɾ��δ��ֵ������ȡֵ��Χ�в�����Լ����ֵ�������DWO
				if (FC(board, domain, level + 1, n))
					//δ��⵽DWO�����������������һ��λ��
					return true;
			//��⵽DWO���򽫸�λ�õĻʺ��Ƴ�
			board[level][i] = 0;
			//��ԭdomain����
			FCCheck(domain, level, i, n, true);
		}
	}
	//δ�ҵ�������ûʺ��λ�ã�����false
	return false;
}

void forwardchecking(int board[30][30], int n) {
	//����domain���󣬲�������λ�õ�ֵ��ʼ��Ϊ0
	int domain[30][30];
	memset(domain, 0, sizeof(domain));
	//������ǰ����㷨
	FC(board, domain, 0, n);
}

int main() {
	int board[30][30], n;
	memset(board, 0, sizeof(board));
	cout << "������N��";
	cin >> n;

	double start_time = clock();
	backtracking(board, 0, n);
	double end_time = clock();
	cout << "Backtracking:" << endl
		<< "��ʱ��" << end_time - start_time << "ms" << endl
		<< "����һ���⣺\n";
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j)
			if (board[i][j])
				cout << 'W';
			else
				cout << 'o';
		cout << endl;
	}
	cout << endl;

	memset(board, 0, sizeof(board));
	start_time = clock();
	forwardchecking(board, n);
	end_time = clock();
	cout << "Forward-checking:" << endl
		<< "��ʱ��" << end_time - start_time << "ms" << endl
		<< "����һ���⣺\n";
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j)
			if (board[i][j])
				cout << 'W';
			else
				cout << 'o';
		cout << endl;
	}
	system("pause");
	return 0;
}