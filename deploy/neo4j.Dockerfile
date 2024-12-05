FROM neo4j:latest
ENV NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD}
ENV NEO4J_PLUGINS='["graph-data-science"]'
ENV NEO4J_dbms_security_procedures_unrestricted=gds.*
ENV NEO4J_dbms_security_procedures_allowlist=gds.*
EXPOSE 7474 7687 7473

