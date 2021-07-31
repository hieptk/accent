import argparse
import os

class movie:
    def __init__(self, genre, director, actor):
        self.genre = genre
        self.director=director
        self.actor=actor


class movie_loader:
    def __init__(self, compressed_id=None):
        self.movie_dict=self.load_movie(compressed_id)
        self.genre_list, self.director_list, self.actor_list=self.load_attribute()


    def load_movie(self, compressed_id):  # map the feature entries in all files, kept in self.features dictionary
        parser = argparse.ArgumentParser(description=''' load movie data''')

        path = os.path.dirname(__file__)
        parser.add_argument('--movie_data_file', type=str, default=os.path.join(path, '../data/auxiliary-mapping.txt'))

        parsed_args, _ = parser.parse_known_args()

        movie_file = parsed_args.movie_data_file
        movie_dict= {}

        fr = open(movie_file, 'r')
        for line in fr:
            lines = line.replace('\n', '').split('|')
            # if len(lines) != 4:
            #     continue
            movie_id = compressed_id.get(int(lines[0]), -1) if compressed_id is not None else int(lines[0])
            if movie_id == -1:
                continue
            genre_list = []
            genres = lines[1].split(',')
            for item in genres:
                genre_list.append(int(item))
            director_list=[]
            directors = lines[2].split(',')
            for item in directors:
                director_list.append(int(item))
            actor_list=[]
            actors = lines[3].split(',')
            for item in actors:
                actor_list.append(int(item))
            new_movie = movie(genre_list, director_list, actor_list)
            movie_dict[movie_id]=new_movie
        fr.close()
        return movie_dict

    def load_attribute(self):  # map the feature entries in all files, kept in self.features dictionary
        parser = argparse.ArgumentParser(description=''' load movie data''')

        path = os.path.dirname(__file__)
        parser.add_argument('--movie_data_file', type=str, default=os.path.join(path, '../data/auxiliary-mapping.txt'))

        parsed_args, _ = parser.parse_known_args()

        movie_file = parsed_args.movie_data_file
        genre_list = []
        director_list=[]
        actor_list=[]
        fr = open(movie_file, 'r')
        for line in fr:
            lines = line.replace('\n', '').split('|')
            # if len(lines) != 4:
            #     continue
            genres = lines[1].split(',')
            for item in genres:
                if int(item) not in genre_list:
                    genre_list.append(int(item))
            directors = lines[2].split(',')
            for item in directors:
                if int(item) not in director_list:
                    director_list.append(int(item))
            actors = lines[3].split(',')
            for item in actors:
                if int(item) not in actor_list:
                    actor_list.append(int(item))
        fr.close()
        return genre_list,director_list,actor_list