import matplotlib.pyplot as plt
import numpy as np

class cube():
    adjacent = {'r': ('w', 'b', 'y', 'g'),
                'b': ('w', 'o', 'y', 'r'),
                'w': ('r', 'g', 'o', 'b'),
                'g': ('w', 'r', 'y', 'o'),
                'o': ('w', 'g', 'y', 'b'),
                'y': ('r', 'b', 'o', 'g')}
    
    faces = {}
    
    def __init__(self, init_state=None):
        if init_state != None:
            for c in init_state.keys():
                self.faces[c] = init_state[c]
        else:
            self.faces['r'] = ['r'] * 8
            self.faces['b'] = ['b'] * 8
            self.faces['w'] = ['w'] * 8
            self.faces['g'] = ['g'] * 8
            self.faces['o'] = ['o'] * 8
            self.faces['y'] = ['y'] * 8
        
    def get_state(self):
        return self.faces

    def get_adj_idx(self, face, adj_face):
        idx = 2 * self.adjacent[face].index(adj_face)
        idx_list = [idx, idx+1, idx+2]
        if idx_list[2] == 8:
            idx_list[2] = 0
        return idx_list
    
    def turn_face(self, face, clockwise):
        if clockwise:
            self.faces[face] = self.faces[face][-2:] + self.faces[face][:-2]
            prev_face = self.adjacent[face][-1]
            idx_list = self.get_adj_idx(prev_face, face)
            temp_row = [self.faces[prev_face][x] for x in idx_list]
            for curr_face in self.adjacent[face]:
                idx_list = self.get_adj_idx(curr_face, face)
                curr_row = [self.faces[curr_face][x] for x in idx_list]
                for x in range(3):
                    self.faces[curr_face][idx_list[x]] = temp_row[x]
                temp_row = curr_row.copy()
        else:
            #self.faces[face] = self.faces[face][2:] + self.faces[face][0:2]
            for i in range(3):
                self.turn_face(face, True)
            
    def display(self):
        print(self.faces)

if __name__ == "__main__":
    cube1 = cube()
    print("initial state:")
    cube1.display()
    cube1.turn_face('r', False)
    cube1.turn_face('w', True)
    cube2 = cube(cube1.get_state())
    print("\nfinal state copied to new cube")
    cube2.display()
