import pandas as pd
import logging

class InclinometryFormat():

    def __init__(self, non_corr_path, list_num):
        self.non_corr_path = non_corr_path
        self.list_num = list_num

    def create_corr_csv(self):
        df = pd.read_excel(self.non_corr_path, sheet_name=self.list_num)
        df = df.iloc[:, [1, 2, 3, 5, 8]]
        df[df.columns[2]] = df[df.columns[2]].fillna(0)
        df[df.columns[3]] = df[df.columns[3]].fillna(0)
        df[df.columns[4]] = df[df.columns[4]].fillna(0)

        df = df.groupby([df.columns[0], df.columns[1]]).agg({df.columns[2]: list, 
                                                             df.columns[4]: list, 
                                                             df.columns[3]: list}).reset_index()
        df.columns = ['Well_name', 'Field', 'MD', 'TVD', 'Angle']
        df['Well_name'] = df['Well_name'].astype('object')
        df['Field'] = df['Field'].astype('object')

        df.to_csv('Format_Incinometry.csv')
        return print('Форматирования Инклинометрии завершено')

path = "E:\\Work\\Well_calc_optim\\Well_optimizer\\ТР.7z\\ТР\\Inclin.xlsx"

test = InclinometryFormat(path, 0)
test.create_corr_csv()