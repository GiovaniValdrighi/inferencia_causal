library(bnlearn)
##questao 1
#criando o dag
dag <- model2network("[A][S][E|A:S][R|E][O|E][T|O:R]")
class(dag)
#plot do dag
graphviz.plot(dag)

##questao 2
arcs(dag)
nodes(dag)
y <- c("A", "S", "E", "O", "R", "T")
for(val in y){
  print(parents(dag, val))
}
for(val in y){
  print(children(dag, val))
}
mb(dag, "A")
mb(dag, "E")
mb(dag, "T")

##questao 3
survey <- read.table("https://raw.githubusercontent.com/robertness/causalML/master/HW/survey.txt",
                     header = TRUE)
head(survey)
bn.mle <- bn.fit(dag, data = survey, method = "mle")
bn.bayes <- bn.fit(dag, data = survey, method = "bayes", iss = 100)
#não sei o que é imaginary sample size

##questão 4
dag2 <- dag
dag2 <- drop.arc(dag2, "E", "O", debug = TRUE)
graphviz.plot(dag2)

bn.mle2 <- bn.fit(dag2, data = survey, method = "mle")
bn.bayes2 <- bn.fit(dag2, data = survey, method = "bayes", iss = 10)
bn.bayes
bn.bayes2

##questão 5
#cpdag do dag original não muda nada
pdag <- cpdag(dag)
graphviz.plot(pdag)

#cpdag adicionando uma nova aresta O -> R
dag3 <- dag
dag3 <- set.arc(dag3, "O", "R")
graphviz.plot(dag3)
graphviz.compare(dag3, cpdag(dag3))

#é o mesmo cpdag se eu inverter O -> R
dag4 <- set.arc(dag, "R", "O")
graphviz.plot(cpdag(dag4))

#calculando os scores
score(dag3, survey, type = "loglik")
score(dag4, survey, type = "loglik")
#o score é o mesmo para todos os dag equivalentes


##questão 
bn.bayes <- bn.fit(dag, survey, method = "bayes")
bn.bayes
