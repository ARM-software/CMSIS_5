@Library("cmsis")

DOCKERINFO = [
    'staging': [
        'registryUrl': 'mcu--docker-staging.eu-west-1.artifactory.aws.arm.com',
        'registryCredentialsId': 'artifactory',
        'k8sPullSecret': 'artifactory-mcu-docker-staging',
        'namespace': 'mcu--docker-staging',
        'image': 'cmsis/linux',
        'label': "${JENKINS_ENV}-${JOB_BASE_NAME}-${BUILD_NUMBER}"
    ],
    'production': [
        'registryUrl': 'mcu--docker.eu-west-1.artifactory.aws.arm.com',
        'registryCredentialsId': 'artifactory',
        'namespace': 'mcu--docker',
        'k8sPullSecret': 'artifactory-mcu-docker',
        'image': 'cmsis/linux',
        'label': 'latest'
    ]
]

dockerinfo = DOCKERINFO['production']

isPrecommit = (JOB_BASE_NAME == 'pre_commit')
isPostcommit = (JOB_BASE_NAME == 'post_commit')
isNightly = (JOB_BASE_NAME == 'nightly')

patternGlobal = [
    '^Jenkinsfile'
]

patternDocker = [
    '^docker/.*'
]

patternCoreM = [
    '^CMSIS/Core/Include/.*',
    '^Device/ARM/ARMCM.*'
]

patternCoreA = [
    '^CMSIS/Core_A/Include/.*',
    '^Device/ARM/ARMCA.*'
]

patternCoreValidation = [
    '^CMSIS/CoreValidation/.*'
]

CONFIGURATIONS = [
    'pre_commit': [
        'mdevices': ['CM0', 'CM3', 'CM4FP', 'CM7DP', 'CM23', 'CM33NS', 'CM35PS'],
        'adevices': ['CA7', 'CA9neon'],
        'devices' : [],
        'configs' : [
            'AC5': ['low', 'tiny'],
            'AC6': ['low', 'tiny'],
            'AC6LTM': ['low', 'tiny'],
            'GCC': ['low', 'tiny']
        ]
    ],
    'post_commit': [
        'devices' : ['CM0', 'CM0plus', 'CM3', 'CM4', 'CM4FP', 'CM7', 'CM7SP', 'CM7DP',
             'CM23', 'CM23S', 'CM23NS', 'CM33', 'CM33S', 'CM33NS',
             'CM35P', 'CM35PS', 'CM35PNS',
             'CA5', 'CA5neon', 'CA7', 'CA7neon', 'CA9', 'CA9neon'],
        'configs' : [
            'AC5': ['low', 'tiny'],
            'AC6': ['low', 'tiny'],
            'AC6LTM': ['low', 'tiny'],
            'GCC': ['low', 'tiny']
        ]
    ],
    'nightly':[
        'devices' : ['CM0', 'CM0plus', 'CM3', 'CM4', 'CM4FP', 'CM7', 'CM7SP', 'CM7DP',
                     'CM23', 'CM23S', 'CM23NS', 'CM33', 'CM33S', 'CM33NS',
                     'CM35P', 'CM35PS', 'CM35PNS',
                     'CA5', 'CA5neon', 'CA7', 'CA7neon', 'CA9', 'CA9neon'],
        'configs' : [
            'AC5': ['low', 'mid', 'high', 'size', 'tiny'],
            'AC6': ['low', 'mid', 'high', 'size', 'tiny'],
            'AC6LTM': ['low', 'mid', 'high', 'size', 'tiny'],
            'GCC': ['low', 'mid', 'high', 'size', 'tiny']
        ]
    ]
]
CONFIGURATION = CONFIGURATIONS[JOB_BASE_NAME]

// ---- PIPELINE CODE ----

def getChangeset() {
    def fileset = sh encoding: 'UTF-8', label: '', returnStdout: true, script: 'git diff --name-only HEAD~1..HEAD'
    return fileset.split('\n')
}

def fileSetMatches(fileset, patternset) {
    return patternset.any { p ->
        fileset.any{ f -> f ==~ p }
    }
}

FORCE_BUILD = false
DOCKER_BUILD = true
CORE_VALIDATION = true
COMMIT = null
VERSION = null

pipeline {
    agent { label 'master' }
    options {
        timestamps()
        timeout(time: 1, unit: 'HOURS')
        ansiColor('xterm')
        skipDefaultCheckout()
    }
    environment {
        CI_ACCOUNT          = credentials('grasci')
        ARTIFACTORY         = credentials('artifactory')
        USER                = "${CI_ACCOUNT_USR}"
        PASS                = "${CI_ACCOUNT_PSW}"
        ARTIFACTORY_API_KEY = "${ARTIFACTORY_PSW}"
    }
    stages {
        stage('Checkout') {
            steps {
                script {
                    COMMIT = checkoutScmWithRetry(3)
                    echo "COMMIT: ${COMMIT}"
                    VERSION = (sh(returnStdout: true, script: 'git describe --always')).trim()
                    echo "VERSION: '${VERSION}'"
                }

                dir('docker') {
                    stash name: 'dockerfile', includes: '**'
                }
            }
        }

        stage('Analyse') {
            when {
                expression { return isPrecommit || isPostcommit }
                beforeOptions true
            }
            steps {
                script {
                    def fileset = changeset
                    def hasGlobal = fileSetMatches(fileset, patternGlobal)
                    def hasDocker = fileSetMatches(fileset, patternDocker)
                    def hasCoreM = fileSetMatches(fileset, patternCoreM)
                    def hasCoreA = fileSetMatches(fileset, patternCoreA)
                    def hasCoreValidation = fileSetMatches(fileset, patternCoreValidation)

echo """Change analysis:
- hasGlobal = ${hasGlobal}
- hasDocker = ${hasDocker}
- hasCoreM = ${hasCoreM}
- hasCoreA = ${hasCoreA}
- hasCoreValidation = ${hasCoreValidation}
"""

                    if (isPrecommit) {
                        if (hasGlobal || hasDocker || hasCoreM || hasCoreValidation) {
                            CONFIGURATION['devices'] += CONFIGURATION['mdevices']
                        }
                        if (hasGlobal || hasDocker || hasCoreA || hasCoreValidation) {
                            CONFIGURATION['devices'] += CONFIGURATION['adevices']
                        }
                    }

                    DOCKER_BUILD &= hasDocker
                    CORE_VALIDATION &= hasGlobal || hasDocker || hasCoreM || hasCoreA || hasCoreValidation

echo """Stage schedule:
- DOCKER_BUILD = ${DOCKER_BUILD}
- CORE_VALIDATION = ${CORE_VALIDATION}
"""
                }
            }
        }

        stage('Docker Lint') {
            when {
                expression { return DOCKER_BUILD }
                beforeOptions true
            }
            agent {
                kubernetes {
                    defaultContainer 'hadolint'
                    slaveConnectTimeout 600
                    yaml """\
                        apiVersion: v1
                        kind: Pod
                        securityContext:
                          runAsUser: 1000
                          runAsGroup: 1000
                        spec:
                          imagePullSecrets:
                            - name: artifactory-mcu-docker
                          securityContext:
                            runAsUser: 1000
                            runAsGroup: 1000
                          containers:
                            - name: hadolint
                              image: mcu--docker.eu-west-1.artifactory.aws.arm.com/hadolint/hadolint:v1.19.0-alpine
                              alwaysPullImage: true
                              imagePullPolicy: Always
                              command:
                                - sleep
                              args:
                                - infinity
                              resources:
                                requests:
                                  cpu: 2
                                  memory: 2Gi
                        """.stripIndent()
                }
            }
            steps {
                dir('docker') {
                    unstash 'dockerfile'

                    sh 'hadolint --format json dockerfile | tee hadolint.log'

                    recordIssues tools: [hadoLint(id: 'hadolint', pattern: 'hadolint.log')],
                                 qualityGates: [[threshold: 1, type: 'DELTA', unstable: true]],
                                 referenceJobName: 'nightly', ignoreQualityGate: true
                }
            }
        }

        stage('Docker Build') {
            when {
                expression { return (isPrecommit || isPostcommit) && DOCKER_BUILD }
                beforeOptions true
            }
            agent {
                kubernetes {
                    defaultContainer 'docker-dind'
                    slaveConnectTimeout 600
                    yaml """\
                        apiVersion: v1
                        kind: Pod
                        spec:
                          imagePullSecrets:
                            - name: artifactory-mcu-docker
                          containers:
                            - name: docker-dind
                              image: docker:dind
                              securityContext:
                                privileged: true
                              volumeMounts:
                                - name: dind-storage
                                  mountPath: /var/lib/docker
                          volumes:
                            - name: dind-storage
                              emptyDir: {}
                        """.stripIndent()
                }
            }
            steps {
                sh('apk add bash curl git')
                script {
                    dir('docker') {
                        unstash 'dockerfile'
                    
                        dockerinfo = DOCKERINFO['staging']
                        withCredentials([sshUserPrivateKey(credentialsId: 'grasci_with_pk',
                                keyFileVariable: 'grasciPk',
                                passphraseVariable: '',
                                usernameVariable: 'grasciUsername')]) {
                            sh("GIT_SSH_COMMAND='ssh -i $grasciPk -o StrictHostKeyChecking=no' ./getDependencies.sh")
                        }
                        docker.withRegistry("https://${dockerinfo['registryUrl']}", dockerinfo['registryCredentialsId']) {
                            def image = docker.build("${dockerinfo['registryUrl']}/${dockerinfo['image']}:${dockerinfo['label']}", "--build-arg DOCKER_REGISTRY=${dockerinfo['registryUrl']} .")
                            image.push()
                        }
                    }
                }
            }
        }

        stage('CoreValidation') {
            when {
                expression { return CORE_VALIDATION }
                beforeOptions true
            }
            matrix {
                axes {
                    axis {
                      name 'DEVICE'
                      values 'CM0', 'CM0plus', 'CM3', 'CM4', 'CM4FP', 'CM7', 'CM7SP', 'CM7DP',
                             'CM23', 'CM23S', 'CM23NS', 'CM33', 'CM33S', 'CM33NS',
                             'CM35P', 'CM35PS', 'CM35PNS',
                             'CA5', 'CA5neon', 'CA7', 'CA7neon', 'CA9', 'CA9neon'
                    }
                }
                stages {
                    stage('Test') {
                        when {
                            expression { return DEVICE in CONFIGURATION['devices'] }
                            beforeOptions true
                        }
                        agent {
                            kubernetes {
                                defaultContainer 'cmsis'
                                slaveConnectTimeout 600
                                yaml """\
                                    apiVersion: v1
                                    kind: Pod
                                    spec:
                                      imagePullSecrets:
                                        - name: ${dockerinfo['k8sPullSecret']}
                                      securityContext:
                                        runAsUser: 1000
                                        runAsGroup: 1000
                                      containers:
                                        - name: cmsis
                                          image: ${dockerinfo['registryUrl']}/${dockerinfo['image']}:${dockerinfo['label']}
                                          alwaysPullImage: true
                                          imagePullPolicy: Always
                                          command:
                                            - sleep
                                          args:
                                            - infinity
                                          resources:
                                            requests:
                                              cpu: 2
                                              memory: 2Gi
                                    """.stripIndent()
                            }
                        }
                        steps {
                            checkoutScmWithRetry(3)
                            dir('CMSIS/CoreValidation/Tests') {
                                script {
                                    CONFIGURATION['configs'].each { COMPILER, OPTS ->
                                        tee("CV_${COMPILER}_${DEVICE}.log") {
                                            sh "python3 build.py -d ${DEVICE} -c ${COMPILER} -o ${OPTS.join(' -o ')} build run"
                                        }
                                    }
                                }

                                archiveArtifacts artifacts: "CoreValidation_*.zip", allowEmptyArchive: true
                                stash name: "CV_${DEVICE}", includes: '*.log, *.junit'
                            }
                        }
                    }
                }
            }
        }

        stage('Results') {
            steps {
                dir('results') {
                    deleteDir()
                    script {
                        CONFIGURATION['devices'].each { unstash "CV_${it}" }
                    }

                    recordIssues tools: [clang(id: 'AC6', name: 'Arm Compiler 6', pattern: 'CV_AC6_*.log'),
                                         clang(id: 'AC6LTM', name: 'Arm Compiler 6 LTM', pattern: 'CV_AC6LTM_*.log')],
                                 qualityGates: [[threshold: 1, type: 'DELTA', unstable: true]],
                                 referenceJobName: 'nightly', ignoreQualityGate: true
                    xunit([
                        JUnit(pattern: 'corevalidation_*.junit', failIfNotNew: false, skipNoTestFiles: true)
                    ])
                }

            }
        }

        stage('Docker Promote') {
            when {
                expression { return isPostcommit && DOCKER_BUILD }
                beforeOptions true
            }
            agent {
                kubernetes {
                    defaultContainer 'docker-dind'
                    slaveConnectTimeout 600
                    yaml """\
                        apiVersion: v1
                        kind: Pod
                        spec:
                          imagePullSecrets:
                            - name: artifactory-mcu-docker
                          containers:
                            - name: docker-dind
                              image: docker:dind
                              securityContext:
                                privileged: true
                              volumeMounts:
                                - name: dind-storage
                                  mountPath: /var/lib/docker
                          volumes:
                            - name: dind-storage
                              emptyDir: {}
                        """.stripIndent()
                }
            }
            steps {
                script {
                    String postCommitTag = "${dockerinfo['registryUrl']}/${dockerinfo['image']}:${dockerinfo['label']}"
                    String prodCommitTag = "${DOCKERINFO['production']['registryUrl']}/${DOCKERINFO['production']['image']}:${DOCKERINFO['production']['label']}"

                    // Pull & retag Docker Staging Container to Production
                    docker.withRegistry("https://${dockerinfo['registryUrl']}", dockerinfo['registryCredentialsId']) {
                        def image = docker.image("$postCommitTag")
                        image.pull()
                        sh "docker tag $postCommitTag $prodCommitTag"
                    }
                    // Push to Docker Production
                    docker.withRegistry("https://${DOCKERINFO['production']['registryUrl']}", DOCKERINFO['production']['registryCredentialsId']) {
                        def image = docker.image("$prodCommitTag")
                        image.push()
                    }
                }
            }
        }
    }
}
