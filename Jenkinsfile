@Library("cmsis")
import com.arm.dsg.cmsis.jenkins.ArtifactoryHelper

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
ALPINE_VERSION = '3.15'
HADOLINT_VERSION = '2.6.0-alpine'

dockerinfo = DOCKERINFO['production']

isPrecommit = (JOB_BASE_NAME == 'pre_commit')
isPostcommit = (JOB_BASE_NAME == 'post_commit')
isNightly = (JOB_BASE_NAME == 'nightly')
isRelease = (JOB_BASE_NAME == 'release')

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
    'pre_commit' : [ "--pairwise" ],
    'post_commit' : [ "--pairwise" ],
    'nightly' : [ ],
    'release' : [ ]
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
DOCKER_BUILD = isPrecommit || isPostcommit || isNightly
CORE_VALIDATION = isPrecommit || isPostcommit || isNightly
COMMIT = null
VERSION = null

artifactory = new ArtifactoryHelper(this)

pipeline {
    agent none
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
            agent {
                kubernetes {
                    defaultContainer 'generic'
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
                            - name: generic
                              image: mcu--docker.eu-west-1.artifactory.aws.arm.com/alpine:${ALPINE_VERSION}
                              command:
                                - sleep
                              args:
                                - infinity
                        """.stripIndent()
                }
            }
            steps {
                script {
                    COMMIT = checkoutScmWithRetry(3)
                    echo "COMMIT: ${COMMIT}"
                    VERSION = (sh(returnStdout: true, script: 'git describe --tags --always')).trim()
                    echo "VERSION: '${VERSION}'"
                }

                stash name: 'dockerfile', includes: 'docker/**'
            }
        }

        stage('Analyse') {
            agent {
                kubernetes {
                    defaultContainer 'generic'
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
                            - name: generic
                              image: mcu--docker.eu-west-1.artifactory.aws.arm.com/alpine:${ALPINE_VERSION}
                              command:
                                - sleep
                              args:
                                - infinity

                        """.stripIndent()
                }
            }
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
                    """.stripIndent()

                    if (isPrecommit) {
                        if (hasGlobal || hasDocker || hasCoreM || hasCoreValidation) {
                            CONFIGURATION += ["--device CM*"]
                        }
                        if (hasGlobal || hasDocker || hasCoreA || hasCoreValidation) {
                            CONFIGURATION += ["--device CA*"]
                        }
                    }

                    DOCKER_BUILD &= hasDocker
                    CORE_VALIDATION &= hasGlobal || hasDocker || hasCoreM || hasCoreA || hasCoreValidation

                    echo """Stage schedule:
                     - DOCKER_BUILD = ${DOCKER_BUILD}
                     - CORE_VALIDATION = ${CORE_VALIDATION}
                    """.stripIndent()
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
                              image: mcu--docker.eu-west-1.artifactory.aws.arm.com/hadolint/hadolint:${HADOLINT_VERSION}
                              alwaysPullImage: true
                              imagePullPolicy: Always
                              command:
                                - sleep
                              args:
                                - infinity
                              resources:
                                requests:
                                  cpu: 900m
                                  memory: 3Gi
                        """.stripIndent()
                }
            }
            steps {
                unstash 'dockerfile'

                sh 'hadolint --format json docker/dockerfile* | tee hadolint.log'

                recordIssues tools: [hadoLint(id: 'hadolint', pattern: 'hadolint.log')],
                             qualityGates: [[threshold: 1, type: 'DELTA', unstable: true]],
                             referenceJobName: 'nightly', ignoreQualityGate: true
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
                              image: mirrors--dockerhub.eu-west-1.artifactory.aws.arm.com/docker:dind
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
                    unstash 'dockerfile'

                    dir('docker') {
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

        stage('Pack') {
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
                                  cpu: 900m
                                  memory: 3Gi
                        """.stripIndent()
                }
            }
            steps {
                checkoutScmWithRetry(3)
                sh('./CMSIS/Utilities/fetch_devtools.sh')
                sh('./CMSIS/RTOS/RTX/LIB/fetch_libs.sh')
                sh('./CMSIS/RTOS2/RTX/Library/fetch_libs.sh')

                tee('doxygen.log') {
                    sh('./CMSIS/DoxyGen/gen_doc.sh')
                }
                sh('./CMSIS/Utilities/gen_pack.sh')

                archiveArtifacts artifacts: 'output/ARM.CMSIS.*.pack', allowEmptyArchive: true
                stash name: 'pack', includes: 'output/ARM.CMSIS.*.pack'

                recordIssues tools: [doxygen(id: 'DOXYGEN', name: 'Doxygen', pattern: 'doxygen.log')],
                             qualityGates: [[threshold: 1, type: 'DELTA', unstable: true]],
                             referenceJobName: 'nightly', ignoreQualityGate: true
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
                      name 'SLICE'
                      values 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                    }
                }
                stages {
                    stage('Test') {
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
                                              cpu: 900m
                                              memory: 3Gi
                                    """.stripIndent()
                            }
                        }
                        steps {
                            checkoutScmWithRetry(3)
                            dir('CMSIS/CoreValidation/Tests') {
                                sh "pip install -e git+https://github.com/energy6/python-matrix-runner#egg=python-matrix-runner"
                                sh "python3 build.py ${CONFIGURATION.join(' ')} --slice ${SLICE}/10 build run"
                                archiveArtifacts artifacts: 'CoreValidation_*.zip', allowEmptyArchive: true
                                stash name: "CV_${SLICE}", includes: '*.log, *.junit'
                            }
                        }
                    }
                }
            }
        }

        stage('Results') {
            agent {
                kubernetes {
                    defaultContainer 'generic'
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
                            - name: generic
                              image: mcu--docker.eu-west-1.artifactory.aws.arm.com/alpine:${ALPINE_VERSION}
                              command:
                                - sleep
                              args:
                                - infinity
                        """.stripIndent()
                }
            }
            when {
                expression { return CORE_VALIDATION }
                beforeOptions true
            }
            steps {
                dir('results') {
                    deleteDir()
                    script {
                        (1..10).each { unstash "CV_${it}" }
                    }

                    recordIssues tools: [armCc(id: 'AC5', name: 'Arm Compiler 5', pattern: 'CV_AC5_*.log'),
                                         clang(id: 'AC6', name: 'Arm Compiler 6', pattern: 'CV_AC6_*.log'),
                                         clang(id: 'AC6LTM', name: 'Arm Compiler 6 LTM', pattern: 'CV_AC6LTM_*.log'),
                                         gcc(id: 'GCC', name: 'GNU Compiler', pattern: 'CV_GCC_*.log')],
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
                              image: mirrors--dockerhub.eu-west-1.artifactory.aws.arm.com/docker:dind
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

        stage('Release Promote') {
            agent {
                kubernetes {
                    defaultContainer 'generic'
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
                            - name: generic
                              image: mcu--docker.eu-west-1.artifactory.aws.arm.com/alpine:${ALPINE_VERSION}
                              command:
                                - sleep
                              args:
                                - infinity
                        """.stripIndent()
                }
            }
            when {
                expression { return isRelease }
                beforeOptions true
            }
            steps {
                unstash name: 'pack'
                dir('output') {
                    script {
                        artifactory.upload pattern: 'ARM.CMSIS.*.pack',
                                           target: "mcu.promoted/CMSIS_5/${VERSION}/",
                                           props: "GIT_COMMIT=${COMMIT['GIT_COMMIT']}"
                    }
                    withCredentials([string(credentialsId: 'grasci_github', variable: 'ghtoken')]) {
                        sh """
                            curl -XPOST \
                                -H "Authorization:token ${ghtoken}" \
                                -H "Content-Type:application/octet-stream" \
                                --data-binary @ARM.CMSIS.${VERSION}.pack \
                                https://uploads.github.com/repos/ARM-software/CMSIS_5/releases/${VERSION}/assets?name=ARM.CMSIS.${VERSION}.pack
                        """
                    }
                }
            }
        }
    }
}
