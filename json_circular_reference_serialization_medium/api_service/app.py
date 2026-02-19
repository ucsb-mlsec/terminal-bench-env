#!/usr/bin/env python3

from flask import Flask, jsonify
import json

app = Flask(__name__)

class Team:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.members = []
    
    def add_member(self, member):
        self.members.append(member)

class Member:
    def __init__(self, id, name, role):
        self.id = id
        self.name = name
        self.role = role
        self.projects = []
    
    def add_project(self, project):
        self.projects.append(project)

class Project:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.members = []
    
    def add_member(self, member):
        self.members.append(member)

# Create sample data with circular references
team1 = Team(1, 'Development Team A')

# Create members
alice = Member(1, 'Alice', 'Developer')
bob = Member(2, 'Bob', 'Designer')
charlie = Member(3, 'Charlie', 'Manager')

# Create projects
project_x = Project(1, 'Project X')
project_y = Project(2, 'Project Y')

# Create circular references
# Members are assigned to projects
alice.add_project(project_x)
alice.add_project(project_y)
bob.add_project(project_x)
charlie.add_project(project_y)

# Projects reference back to members
project_x.add_member(alice)
project_x.add_member(bob)
project_y.add_member(alice)
project_y.add_member(charlie)

# Add members to team
team1.add_member(alice)
team1.add_member(bob)
team1.add_member(charlie)

# Store teams
teams = {1: team1}

@app.route('/api/team/<int:team_id>')
def get_team(team_id):
    team = teams.get(team_id)
    if not team:
        return jsonify({'error': 'Team not found'}), 404
    
    # This will cause circular reference error
    return jsonify(team)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)