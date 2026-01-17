pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage('Cloning Repository'){
            steps{
                echo 'Cloning the repository...'
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: '29081395-60db-4b6c-ac89-07b53967d992', url: 'https://github.com/csergey88/hotel-reservation']])
            }
        }
        stage('Setup Environment'){
            steps{
                echo 'Setting up the environment...'
                sh '''
                python3 -m venv $VENV_DIR
                . $VENV_DIR/bin/activate
                pip install --upgrade pip
                pip install -e .
                '''
            }
        }
        stage('Deploy'){
            steps{
                echo 'Deploying the application...'
            }
        }
    }
}
