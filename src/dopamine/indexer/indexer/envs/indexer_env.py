import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import re
import sql_metadata
import os
import psycopg2
import yaml

SCHEMA = "resources/schema.sql"
FOLDER = "dummy"
QUERIES_DIRECTORY = f"{FOLDER}/queries/"

class IndexerEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.env = self
    self.column_name_to_matrix_col = {}
    self.matrix_col_to_column_name = {}
    self.connection = psycopg2.connect("dbname=tpch")

    self.build_columns_dictionary()

    self.MAX_INDEXES = 3

    self.columns = len(self.matrix_col_to_column_name.keys())

    self.rows = len(os.listdir(os.path.join(os.path.dirname(__file__), QUERIES_DIRECTORY))) + 2
    self.N_DISCRETE_ACTIONS = self.columns * 2
    self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)

    self.initial_cost = self.calculate_cost(True)
    self.current_cost = self.initial_cost

    self.state = None
    self.reset()

  def step(self, action):
    assert self.action_space.contains(action), "%r invalid"%(action)
    print('STEP && action: ' + str(action))
    state = self.state

    done = self.should_stop(action, state)
    if not done:
      if action >= self.columns:
        column = self.column_name_to_matrix_col[action % self.columns]
        self.remove_virtual_index(column)
        state[0][column] = 0
      else:
        self.add_virtual_index(self.matrix_col_to_column_name[action])
        state[0][action] = 1
      reward = 1#self.calculate_reward()
    else:
      reward = 0

    self.state = state
    self.current_cost = self.calculate_cost()

    print('STEP done!')
    return self.state, reward, done, {}

  def reset(self):
    indexes = np.zeros((1, self.columns))
    matrix = np.concatenate((indexes, np.ones((self.rows - 1, self.columns))))

    self.fill_in_selectivities(self.calculate_selectivities(matrix), matrix)
    self.state = matrix
    self.current_cost = self.initial_cost

    cursor = self.connection.cursor()
    cursor.execute("select * from hypopg_reset();")
    cursor.close()
    return self.state

  def add_virtual_index(self, column):
    cursor = self.connection.cursor()
    cursor.execute(f"select * from hypopg_create_index('create index on lineitem ({column})');")
    cursor.close()

  def remove_virtual_index(self, column):
    cursor = self.connection.cursor()
    cursor.execute(f"select * from hypopg_drop_index('create index on tpch ({column})');")
    cursor.close()

  def calculate_cost(self, initial = False):
    cost = 0
    cursor = self.connection.cursor()

    if initial:
      costs = yaml.load(open(os.path.join(os.path.dirname(__file__), FOLDER + '/costs.yml'), "r").read())
      cost = sum(costs.values())
    else:
      queries = np.sort(os.listdir(os.path.join(os.path.dirname(__file__), QUERIES_DIRECTORY)))
      for index, query_file in enumerate(queries):
        cursor.execute("explain analyze " + open(os.path.join(os.path.dirname(__file__), QUERIES_DIRECTORY + query_file), "r").read())
        results = cursor.fetchall()
        execution_time = float(results[-1][0].split()[2])
        cost += execution_time

    cursor.close()
    return cost

  def should_stop(self, action, matrix):
    #TODO: check if there is an index for that column already
    if action >= self.N_DISCRETE_ACTIONS:
      print('Invalid column')
      return True
    elif np.sum(np.array(matrix)[:, action % self.columns]) == 0:
      print('There\'s no query with that column :/,')
    else:
      if action >= self.columns and matrix[0][action % self.columns] == 0:
        print('There isn\'t an index on that column so we can\'t drop it ¯\_(ツ)_/¯')
        return True
      elif action < self.columns and self.MAX_INDEXES == np.array(matrix)[0].sum():
        print('You can\' create more indexes')
        return True
      else:
        return False

  def read_schema(self):
    schema = open(os.path.join(os.path.dirname(__file__), SCHEMA), "r").read()
    return schema.replace("\n", " ").replace("\r", "").split(";")

  def table_definitions(self):
    return filter(lambda line: "CREATE TABLE" in line, self.read_schema())

  def parse_columns_from_table_definitions(self, table_definition):
    column_definitions = table_definition[table_definition.find("(") + 1: -1].strip()
    column_dictionary = {}

    for column in column_definitions.split(","):
      matches = re.match(r"(\w+) (\w+)(\((\d+)\)| \w+\((\d+)\)| \w+ \w+)?", column.strip())
      column_name = matches.group(1)
      if matches.group(2) == "integer":
        column_dictionary[column_name] = {"type": "integer"}
      elif matches.group(2) == "bigint":
        column_dictionary[column_name] = {"type": "bigint"}
      elif matches.group(2) == "numeric":
        column_dictionary[column_name] = {"type": "numeric"}
      elif matches.group(2) == "character":
        column_dictionary[column_name] = {"type": "character", "size": (matches.group(5) or matches.group(4))}
      elif matches.group(2) == "date":
        column_dictionary[column_name] = {"type": "date"}
      else:
        raise ValueError("invalid column type")

    return column_dictionary

  def build_columns_dictionary(self):
    columns = 0
    for table in self.table_definitions():
      if not 'lineitem' in table:
        continue
      columns_dic = self.parse_columns_from_table_definitions(table)
      for index, column in enumerate(columns_dic.keys()):
        self.column_name_to_matrix_col[column] = columns + index
        self.matrix_col_to_column_name[columns + index] = column
      columns += len(columns_dic.keys())

  def fill_in_selectivities(self, selectivities, matrix):
    for query, values in selectivities.items():
      for column, sel in values.items():
        matrix[query + 1][self.column_name_to_matrix_col[column]] = sel

  def calculate_selectivities(self, matrix):
    cursor = self.connection.cursor()
    selectivities = {}

    queries = np.sort(os.listdir(os.path.join(os.path.dirname(__file__), QUERIES_DIRECTORY)))
    for index, query_file in enumerate(queries):
      query = query_file.split(".")[0]
      # check if selectivity is stored, if not run the query and get it.
      if not os.path.exists(os.path.join(os.path.dirname(__file__), f"./{FOLDER}/selectivities/" + query + ".yml")):
        # TODO: fix this so that it stores the values correctly
        print("Selectivity for query: " + query + " missing, running the query now...")
        cursor.execute(open(os.path.join(os.path.dirname(__file__), QUERIES_DIRECTORY + query_file), "r").read())
        matrix[index][1] = cursor.rowcount

        file = open(os.path.join(os.path.dirname(__file__), f"./{FOLDER}/selectivities/" + query + ".yml"), "w")
        file.write(str(cursor.rowcount))
        file.close()
        print("done")
      else:
        columns = yaml.load(open(os.path.join(os.path.dirname(__file__), f"./{FOLDER}/selectivities/" + query + ".yml")))
        if not columns is None:
          query_selectivity = {}
          for column, sel in columns.items():
            query_selectivity[column] = sel['scoped']/sel['count']
          selectivities[index] = query_selectivity
    cursor.close()
    return selectivities

  def calculate_reward(self):
    return max(self.initial_cost/self.current_cost - 1, 0)
